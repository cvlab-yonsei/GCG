from mmengine.config import read_base

with read_base():
    from .minilongbench_qmsum_gen_d33331 import MiniLongBench_qmsum_datasets  # noqa: F401, F403
