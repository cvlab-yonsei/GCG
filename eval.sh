#!/usr/bin/env bash
# Evaluate LLaDA-8B-Base with and without GCG on the five benchmarks.
#
# Usage:
#   bash eval.sh                       # all benchmarks
#   bash eval.sh ruler gsm8k_cot       # a subset
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash eval.sh
#
# Benchmarks: ruler, minilongbench, gsm8k_cot, humaneval, mbpp
# All visible GPUs are used. Results are written to outputs/.
set -euo pipefail
cd "$(dirname "$0")"

LLADA_PATH=${LLADA_PATH:-GSAI-ML/LLaDA-8B-Base}
LLADA_REVISION=${LLADA_REVISION:-0f2787f2d87eac5eed8a087d5ecd24277e6255b2}
GCG="enhance_pkl=assets/com_llada_8b_base.pkl,enhance_layers=4-8-15-17-18-24-28-30,enhance_alpha=0.3,enhance_pct=0.25"

export HF_ALLOW_CODE_EVAL=1   # HumanEval / MBPP execute model-generated code.

# Long-context benchmarks (OpenCompass): base and GCG are both defined in the config.
run_opencompass() {
    local CONFIG=$1
    python run.py "${CONFIG}" --dump-eval-details
}

# Short-context benchmarks (lm-evaluation-harness): base, then GCG.
run_lmeval() {
    local TASK=$1 GEN_ARGS=$2 EXTRA=$3
    local NPROC LAUNCH
    NPROC=$(python -c "import torch; print(torch.cuda.device_count())")
    if [ "${NPROC}" -gt 1 ]; then
        LAUNCH=(accelerate launch --multi_gpu --num_processes "${NPROC}" --main_process_port "${PORT:-29500}")
    else
        LAUNCH=(python)
    fi
    for VARIANT in base gcg; do
        local MODEL_ARGS="model_path=${LLADA_PATH},revision=${LLADA_REVISION},${GEN_ARGS}"
        [ "${VARIANT}" = gcg ] && MODEL_ARGS="${MODEL_ARGS},${GCG}"
        echo "=== ${TASK} / ${VARIANT} ==="
        "${LAUNCH[@]}" short_context/eval_llada.py --tasks "${TASK}" --model llada_gcg ${EXTRA} \
            --output_path "outputs/${TASK}/${VARIANT}" --log_samples --model_args "${MODEL_ARGS}"
    done
}

BENCHMARKS=("$@")
[ ${#BENCHMARKS[@]} -eq 0 ] && BENCHMARKS=(ruler minilongbench gsm8k_cot humaneval mbpp)

for B in "${BENCHMARKS[@]}"; do
    case "${B}" in
        ruler)
            run_opencompass configs/eval_llada_ruler_8k.py ;;
        minilongbench)
            if [ ! -d data/MiniLongBench/data ]; then
                echo "MiniLongBench data not found at data/MiniLongBench/data/. See README." >&2
                exit 1
            fi
            run_opencompass configs/eval_llada_minilongbench.py ;;
        gsm8k_cot)
            run_lmeval gsm8k_cot "gen_length=512,steps=128,block_length=512" "" ;;
        humaneval)
            run_lmeval humaneval "gen_length=256,steps=64,block_length=256" "--confirm_run_unsafe_code" ;;
        mbpp)
            run_lmeval mbpp "gen_length=256,steps=64,block_length=256" "--confirm_run_unsafe_code" ;;
        *)
            echo "Unknown benchmark: ${B} (choose from ruler, minilongbench, gsm8k_cot, humaneval, mbpp)" >&2
            exit 1 ;;
    esac
done
