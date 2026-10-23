"""MiniLongBench (21 tasks) on LLaDA-8B-Base: base vs. GCG.

No RoPE scaling, prompt truncated to 7500 tokens (drop middle), gen length 512,
128 steps, block length 512, greedy decoding.
"""
from mmengine.config import read_base

from opencompass.models import LLaDACausalLM
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.minilongbench.minilongbench import minilongbench_datasets

datasets = minilongbench_datasets

# Run from the repository root (see eval.sh). To use a local copy of the
# model, set LLADA_PATH to its directory and LLADA_REVISION to None.
LLADA_PATH = 'GSAI-ML/LLaDA-8B-Base'
LLADA_REVISION = '0f2787f2d87eac5eed8a087d5ecd24277e6255b2'
COM_PKL = 'assets/com_llada_8b_base.pkl'
LAYERS = '4-8-15-17-18-24-28-30'

common = dict(
    type=LLaDACausalLM,
    path=LLADA_PATH,
    revision=LLADA_REVISION,
    drop_middle=True,
    diffusion_config={'steps': 128, 'block_length': 512},
    seed=2025,
    max_out_len=512,
    max_seq_len=7500,
    batch_size=1,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

models = [
    dict(abbr='llada_8b-minilongbench-base', **common),
    dict(abbr='llada_8b-minilongbench-gcg', enhance_pkl=COM_PKL, enhance_layers=LAYERS,
         enhance_alpha=0.3, enhance_pct=0.25, **common),
]

work_dir = './outputs/minilongbench/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=200, gen_task_coef=15),
    runner=dict(type=LocalRunner, max_num_workers=8, task=dict(type=OpenICLInferTask)),
)
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=8,
                task=dict(type=OpenICLEvalTask, dump_details=True)),
)
