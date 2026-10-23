from mmengine.config import read_base

with read_base():
    from .minilongbench_passage_count_gen_dcdaab import MiniLongBench_passage_count_datasets  # noqa: F401, F403
