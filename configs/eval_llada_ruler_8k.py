"""RULER 8K (CWE / FWE / QA-SQuAD / QA-HotpotQA) on LLaDA-8B-Base: base vs. GCG.

NTK x4, gen length 64, 16 steps, block length 16, greedy decoding.
"""
from mmengine.config import read_base

from opencompass.models import LLaDACausalLM
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.ruler.ruler_cwe_gen import cwe_datasets as cwe
    from opencompass.configs.datasets.ruler.ruler_fwe_gen import fwe_datasets as fwe
    from opencompass.configs.datasets.ruler.ruler_qa_gen import qa_datasets as qa

# Run from the repository root (see eval.sh). To use a local copy of the
# model, set LLADA_PATH to its directory and LLADA_REVISION to None.
LLADA_PATH = 'GSAI-ML/LLaDA-8B-Base'
LLADA_REVISION = '0f2787f2d87eac5eed8a087d5ecd24277e6255b2'
COM_PKL = 'assets/com_llada_8b_base.pkl'
LAYERS = '4-8-15-17-18-24-28-30'

datasets = []
for dataset in sum((cwe, fwe, qa), []):
    tmp = dataset.deepcopy()
    tmp['abbr'] = tmp['abbr'] + '_8k'
    tmp['num_samples'] = 100
    tmp['max_seq_length'] = 1024 * 8
    tmp['tokenizer_model'] = 'gpt-4'
    datasets.append(tmp)

common = dict(
    type=LLaDACausalLM,
    path=LLADA_PATH,
    revision=LLADA_REVISION,
    scaling_config={'scaling_factor': 4},
    diffusion_config={'steps': 16, 'block_length': 16},
    seed=2025,
    max_out_len=64,
    batch_size=1,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

models = [
    dict(abbr='llada_8b-ruler_8k-base', **common),
    dict(abbr='llada_8b-ruler_8k-gcg', enhance_pkl=COM_PKL, enhance_layers=LAYERS,
         enhance_alpha=0.3, enhance_pct=0.25, **common),
]

work_dir = './outputs/ruler_8k/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=200, gen_task_coef=15),
    runner=dict(type=LocalRunner, max_num_workers=8, task=dict(type=OpenICLInferTask)),
)
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=8,
                task=dict(type=OpenICLEvalTask, dump_details=True)),
)
