# Pytorch implementation of Global Context Guidance

This is the pytorch implementation of the paper **Global Context Guidance for Diffusion Large Language Models** (NeurIPS 2026)

For detailed information, please checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/GCG/)]
<!-- [Project Page]() | [arXiv]() -->


## Requirements

Python 3.11 and CUDA 12.1. The pinned versions are in `requirements/llada.txt`; `requirements.txt` also installs this repository in editable mode.

Install dependencies:
```bash
pip install -r requirements.txt
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```


## Dataset

- **RULER**, **GSM8K-CoT**, **HumanEval**, **MBPP**: downloaded automatically on the first run.
- **MiniLongBench**: download from [Hugging Face](https://huggingface.co/datasets/linggm/MiniLongBench) ([official repo](https://github.com/MilkThink-Lab/MiniLongBench)) into `data/MiniLongBench/`:
  ```bash
  huggingface-cli download linggm/MiniLongBench --repo-type dataset --include "data/*" --local-dir data/MiniLongBench
  ```

## Getting Started

`eval.sh` evaluates both the base model and GCG on each benchmark. Results are written to `outputs/`.

```bash
bash eval.sh                       # all benchmarks: ruler, minilongbench, gsm8k_cot, humaneval, mbpp
```


## Acknowledgement
The codebase of this repository is largely borrowed from [OpenCompass](https://github.com/open-compass/opencompass) and [LLaDA](https://github.com/ML-GSAI/LLaDA). Short-context benchmarks are evaluated with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Citation

```bibtex
@inproceedings{jeon2026global,
  title={Global Context Guidance for Diffusion Large Language Models},
  author={Jeon, Jeimin and Ham, Bumsub},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
