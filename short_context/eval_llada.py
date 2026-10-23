"""lm-evaluation-harness model for LLaDA with optional global-head enhancement (GCG).

Supports generation tasks (`generate_until`) such as gsm8k_cot, humaneval and mbpp.
The GCG update and the LLaDA sampler are shared with the OpenCompass wrapper
(`opencompass/models/llada/`).

Usage (see eval.sh):
    accelerate launch short_context/eval_llada.py --tasks gsm8k_cot --model llada_gcg \
        --model_args model_path=GSAI-ML/LLaDA-8B-Base,gen_length=512,steps=128,block_length=512
"""
import random

import accelerate
import numpy as np
import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from opencompass.models.llada.enhance import apply_gcg, load_com_scores
from opencompass.models.llada.llada_generate import generate


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model('llada_gcg')
class LLaDAEvalHarness(LM):
    """LLaDA for lm-eval generation tasks.

    Args:
        model_path: HF model id or local path of LLaDA.
        revision: HF commit to load (pins the remote modeling code).
        gen_length, steps, block_length: sampler settings (greedy decoding).
        enhance_pkl: per-head COM scores; GCG is disabled if empty.
        enhance_layers: target layers, e.g. '4-8-15-17-18-24-28-30'.
        enhance_alpha: GCG strength.
        enhance_pct: fraction of heads used as the global (and local) set.
    """

    def __init__(
        self,
        model_path='GSAI-ML/LLaDA-8B-Base',
        revision=None,
        mask_id=126336,
        batch_size=1,
        gen_length=256,
        steps=64,
        block_length=256,
        enhance_pkl='',
        enhance_layers='',
        enhance_alpha=0.3,
        enhance_pct=0.25,
        device='cuda',
        **kwargs,
    ):
        super().__init__()

        accelerator = accelerate.Accelerator()
        self.accelerator = accelerator if accelerator.num_processes > 1 else None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs['device_map'] = {'': f'{self.accelerator.device}'}

        revision = revision or None
        config = AutoConfig.from_pretrained(model_path, revision=revision, trust_remote_code=True)
        config.flash_attention = True
        self.model = AutoModel.from_pretrained(model_path, config=config, revision=revision,
                                               trust_remote_code=True, torch_dtype=torch.bfloat16,
                                               **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, trust_remote_code=True)
        self.mask_id = mask_id
        self.batch_size = int(batch_size)
        self.gen_length = int(gen_length)
        self.steps = int(steps)
        self.block_length = int(block_length)

        if enhance_pkl:
            com_scores = load_com_scores(enhance_pkl)
            if enhance_layers:
                layer_ids = [int(x) for x in str(enhance_layers).split('-')]
            else:
                layer_ids = sorted(com_scores)
            apply_gcg(self.model, layer_ids, com_scores,
                      pct=float(enhance_pct), alpha=float(enhance_alpha))
            print(f'[GCG] enabled: layers={layer_ids}, alpha={enhance_alpha}, pct={enhance_pct}', flush=True)

    @property
    def rank(self):
        return self._rank if self.accelerator is not None else 0

    @property
    def world_size(self):
        return self._world_size if self.accelerator is not None else 1

    def loglikelihood(self, requests):
        raise NotImplementedError('Only generation tasks are supported.')

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError('Only generation tasks are supported.')

    def generate_until(self, requests: list[Instance]):
        out = []
        for req in tqdm(requests, desc='Generating...', disable=self.rank != 0):
            question, gen_kwargs = req.args
            stop_tokens = gen_kwargs['until']
            prompt = torch.tensor(self.tokenizer(question)['input_ids'], device=self.device)[None]

            generated = generate(self.model, prompt, steps=self.steps, gen_length=self.gen_length,
                                 block_length=self.block_length, temperature=0., cfg_scale=0.,
                                 remasking='low_confidence', mask_id=self.mask_id)

            answer = self.tokenizer.decode(generated[0][prompt.shape[1]:], skip_special_tokens=False)
            for stop_seq in stop_tokens:
                if stop_seq in answer:
                    answer = answer.split(stop_seq)[0]

            # Remove special tokens.
            answer_ids = self.tokenizer(answer)['input_ids']
            out.append(self.tokenizer.decode(answer_ids, skip_special_tokens=True))

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
        return out


if __name__ == '__main__':
    set_seed(1234)
    cli_evaluate()
