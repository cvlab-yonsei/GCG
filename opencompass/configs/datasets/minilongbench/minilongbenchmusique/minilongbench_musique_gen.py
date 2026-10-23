from mmengine.config import read_base

with read_base():
    from .minilongbench_musique_gen_6b3efc import MiniLongBench_musique_datasets  # noqa: F401, F403
