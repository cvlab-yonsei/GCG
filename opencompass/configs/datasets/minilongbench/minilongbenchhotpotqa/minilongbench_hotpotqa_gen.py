from mmengine.config import read_base

with read_base():
    from .minilongbench_hotpotqa_gen_6b3efc import MiniLongBench_hotpotqa_datasets  # noqa: F401, F403
