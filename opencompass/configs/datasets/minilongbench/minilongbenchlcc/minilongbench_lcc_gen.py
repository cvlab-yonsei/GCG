from mmengine.config import read_base

with read_base():
    from .minilongbench_lcc_gen_6ba507 import MiniLongBench_lcc_datasets  # noqa: F401, F403
