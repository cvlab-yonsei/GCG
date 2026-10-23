import os
import random
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from opencompass.models.base import LMTemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger

from ..huggingface_above_v4_33 import HuggingFaceBaseModel
from .llada_generate import generate


def _convert_base_messages(inputs):
    outputs = []
    for _input in inputs:
        if isinstance(_input, str):
            outputs.append(_input)
        else:
            outputs.append(''.join(item['prompt'] for item in _input))
    return outputs


@MODELS.register_module()
class LLaDACausalLM(HuggingFaceBaseModel):
    """LLaDA masked-diffusion LM with optional global-head enhancement (GCG).

    Args:
        path: HF model id or local path of LLaDA.
        revision: HF commit to load (pins the remote modeling code).
        max_seq_len: Maximum prompt length in tokens.
        drop_middle: If the prompt exceeds ``max_seq_len``, keep its first and
            last ``max_seq_len // 2`` tokens (LongBench-style truncation).
        scaling_config: ``{'scaling_factor': s}`` multiplies ``rope_theta`` by
            ``s`` (NTK-aware RoPE scaling).
        diffusion_config: Overrides for :func:`llada_generate.generate`
            (``steps``, ``block_length``, ``temperature``, ...).
        seed: Global random seed.
        enhance_pkl: Per-head COM scores. Enhancement is disabled if None.
        enhance_layers: Target layers, e.g. ``'4-8-15-17-18-24-28-30'``.
        enhance_alpha: Enhancement strength.
        enhance_pct: Fraction of heads used as the global (and local) set.
    """

    def __init__(self,
                 path: str,
                 revision: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 tokenizer_only: bool = False,
                 generation_kwargs: dict = dict(),
                 max_seq_len: Optional[int] = None,
                 pad_token_id: Optional[int] = None,
                 stop_words: Optional[List[str]] = [],
                 drop_middle: bool = False,
                 scaling_config: Optional[dict] = None,
                 diffusion_config: Optional[dict] = None,
                 seed: Optional[int] = None,
                 enhance_pkl: Optional[str] = None,
                 enhance_layers: Optional[str] = None,
                 enhance_alpha: float = 0.3,
                 enhance_pct: float = 0.25,
                 **other_kwargs):

        if seed is not None:
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self.logger = get_logger()
        self.path = path
        self.revision = revision
        self.tokenizer_only = tokenizer_only
        self.template_parser = LMTemplateParser()
        self.max_seq_len = max_seq_len
        self.drop_middle = drop_middle
        if revision is not None and not tokenizer_path:
            tokenizer_kwargs = dict(tokenizer_kwargs, revision=revision)
        self._load_tokenizer(tokenizer_path or path, tokenizer_kwargs, pad_token_id)

        self.scaling_config = scaling_config
        self.enhance_pkl = enhance_pkl
        self.enhance_layers = enhance_layers
        self.enhance_alpha = enhance_alpha
        self.enhance_pct = enhance_pct

        self.diffusion_config = {'steps': 128, 'block_length': 32, 'temperature': 0.,
                                 'cfg_scale': 0., 'remasking': 'low_confidence'}
        if diffusion_config is not None:
            self.diffusion_config.update(diffusion_config)
        self.logger.info(f'diffusion_config: {self.diffusion_config}')

        if not tokenizer_only:
            self._load_model(path)
        self.generation_kwargs = generation_kwargs
        self.stop_words = stop_words

        for k, v in other_kwargs.items():
            if v is not None:
                self.logger.warning(f'Unused argument {k}={v}')

    def _load_model(self, path: str):
        config = AutoConfig.from_pretrained(path, revision=self.revision, trust_remote_code=True)
        config.flash_attention = True

        if self.scaling_config is not None:
            scaling_factor = self.scaling_config.get('scaling_factor', 1)
            config.rope_theta = config.rope_theta * scaling_factor
            self.logger.info(f'NTK scaling: rope_theta={config.rope_theta}')

        self.model = AutoModelForCausalLM.from_pretrained(
            path, config=config, revision=self.revision, device_map='auto',
            torch_dtype=torch.float16, trust_remote_code=True)
        self.model.eval()
        self.model.generation_config.do_sample = False

        if self.enhance_pkl:
            from .enhance import apply_gcg, load_com_scores
            com_scores = load_com_scores(self.enhance_pkl)
            if self.enhance_layers:
                layer_ids = [int(x) for x in self.enhance_layers.split('-')]
            else:
                layer_ids = sorted(com_scores)
            apply_gcg(self.model, layer_ids, com_scores,
                      pct=self.enhance_pct, alpha=self.enhance_alpha)
            self.logger.info(f'GCG enabled: layers={layer_ids}, '
                             f'alpha={self.enhance_alpha}, pct={self.enhance_pct}')

    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: List[str] = [],
                 **kwargs) -> List[str]:
        messages = _convert_base_messages(inputs)

        if self.drop_middle:
            assert len(inputs) == 1
            input_ids = self.tokenizer(inputs, padding=False, truncation=False)['input_ids']
            input_ids = torch.tensor(input_ids)
            if input_ids.shape[-1] > self.max_seq_len:
                input_ids = torch.cat([input_ids[:, : self.max_seq_len // 2],
                                       input_ids[:, - self.max_seq_len // 2:]], dim=-1)
            tokens = {'input_ids': input_ids}
        else:
            tokens = self.tokenizer.batch_encode_plus(
                messages, return_tensors='pt', padding=True, truncation=True,
                add_special_tokens=True, max_length=self.max_seq_len)
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

        stopping_criteria = list(set(stopping_criteria + self.stop_words))

        diffusion_config = self.diffusion_config.copy()
        block_length = diffusion_config['block_length']
        if max_out_len % block_length != 0:
            max_out_len = (max_out_len // block_length + 1) * block_length
        diffusion_config['steps'] = min(diffusion_config['steps'], max_out_len)

        outputs = generate(self.model, tokens['input_ids'],
                           gen_length=max_out_len, **diffusion_config)
        outputs = outputs[:, tokens['input_ids'].shape[1]:]

        decodeds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for stop in stopping_criteria:
            decodeds = [token.split(stop)[0] for token in decodeds]
        return decodeds

    def get_token_len(self, prompt: str, add_special_tokens: bool = True) -> int:
        m = _convert_base_messages([prompt])[0]
        t = self.tokenizer(m, add_special_tokens=add_special_tokens)
        return len(t['input_ids'])
