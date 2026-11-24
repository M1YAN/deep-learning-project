#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

usage() {
    cat <<'USAGE'
Usage:
  bash scripts/run_experiments.sh prepare
  bash scripts/run_experiments.sh {smoke|pilot|course|course_long|course_wide|course_256|course_256_long} [method|all] [seed]
  bash scripts/run_experiments.sh report {smoke|pilot|course|course_long|course_wide|course_256|course_256_long}

Environment:
  PYTHON       Python interpreter (default .venv/bin/python or python3)
  DEVICE       auto, cpu, cuda, or cuda:N (default auto)
  PREP_DEVICE  Text encoding device (default DEVICE)
  GPUS         Comma-separated physical CUDA indices for one job per GPU
  THREADS      PyTorch and preparation CPU threads per process (default 4)
USAGE
}

mode=${1:-}
case "$mode" in
    prepare|smoke|pilot|course|course_long|course_wide|course_256|course_256_long|report) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then
        PYTHON=.venv/bin/python
    else
        PYTHON=python3
    fi
fi
device=${DEVICE:-auto}
threads=${THREADS:-4}
export HF_HOME="$PWD/data/huggingface"
export TOKENIZERS_PARALLELISM=false

if [[ "$mode" == report ]]; then
    [[ $# == 2 ]] || { usage >&2; exit 2; }
    exec "$PYTHON" -m sidrec.report "$2"
fi

gpus=()
if [[ -n "${GPUS:-}" ]]; then
    [[ "$GPUS" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ && "$device" != cpu ]] || {
        echo "GPUS expects a comma-separated list of CUDA indices and a CUDA device" >&2
        exit 2
    }
    IFS=, read -r -a gpus <<< "$GPUS"
    declare -A seen_gpus=()
    for gpu in "${gpus[@]}"; do
        [[ ! -v "seen_gpus[$gpu]" ]] || {
            echo "GPUS contains a duplicate index: $gpu" >&2
            exit 2
        }
        seen_gpus[$gpu]=1
    done
    device=cuda
fi

methods=(sequential global_pop zcr neighbors neighbors_zero)
seeds=(17)
if [[ "$mode" == course* ]]; then
    seeds=(17 42 73)
fi

if [[ "$mode" == prepare ]]; then
    [[ $# == 1 ]] || { usage >&2; exit 2; }
else
    [[ $# -le 3 && ${3:-17} =~ ^[0-9]+$ ]] || { usage >&2; exit 2; }
    case "${2:-all}" in
        all) ;;
        sequential|global_pop|zcr|neighbors|neighbors_zero) methods=("$2") ;;
        *) usage >&2; exit 2 ;;
    esac
    if [[ $# == 3 ]]; then
        seeds=("$3")
    fi
fi

bash scripts/download_beauty.sh
if (( ${#gpus[@]} )); then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    CUDA_VISIBLE_DEVICES="${gpus[0]}" flock data/.prepare.lock \
        "$PYTHON" -m sidrec.prepare --device "${PREP_DEVICE:-cuda}" --threads "$threads"
else
    flock data/.prepare.lock "$PYTHON" -m sidrec.prepare \
        --device "${PREP_DEVICE:-$device}" --threads "$threads"
fi
[[ "$mode" != prepare ]] || exit 0
if [[ "$mode" == course_wide ]]; then
    flock data/.prepare.lock "$PYTHON" -m sidrec.prepare --wide --threads "$threads"
elif [[ "$mode" == course_256* ]]; then
    flock data/.prepare.lock "$PYTHON" -m sidrec.prepare --prefix-256 --threads "$threads"
fi

run_job() {
    local method=$1 seed=$2 log_dir="runs/$mode/$1/$2"
    mkdir -p "$log_dir"
    "$PYTHON" -m sidrec.train "$mode" "$method" "$seed" --device "$device" --threads "$threads" \
        2>&1 | tee -a "$log_dir/train.log"
}

job_methods=()
job_seeds=()
for seed in "${seeds[@]}"; do
    for method in "${methods[@]}"; do
        job_methods+=("$method")
        job_seeds+=("$seed")
    done
done

if (( ${#gpus[@]} )); then
    pids=()
    for ((worker = 0; worker < ${#gpus[@]} && worker < ${#job_methods[@]}; worker++)); do
        (
            export CUDA_VISIBLE_DEVICES="${gpus[worker]}"
            for ((job = worker; job < ${#job_methods[@]}; job += ${#gpus[@]})); do
                run_job "${job_methods[job]}" "${job_seeds[job]}"
            done
        ) &
        pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=$?
    done
    (( status == 0 )) || exit "$status"
else
    for job in "${!job_methods[@]}"; do
        run_job "${job_methods[job]}" "${job_seeds[job]}"
    done
fi

"$PYTHON" -m sidrec.report "$mode"
