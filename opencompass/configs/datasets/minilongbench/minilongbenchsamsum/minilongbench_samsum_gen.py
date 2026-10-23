from mmengine.config import read_base

with read_base():
    from .minilongbench_samsum_gen_f4416d import MiniLongBench_samsum_datasets  # noqa: F401, F403
