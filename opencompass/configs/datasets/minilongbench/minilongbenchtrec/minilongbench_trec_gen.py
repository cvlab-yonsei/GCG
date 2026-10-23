from mmengine.config import read_base

with read_base():
    from .minilongbench_trec_gen_824187 import MiniLongBench_trec_datasets  # noqa: F401, F403
