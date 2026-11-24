#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then
        PYTHON=.venv/bin/python
    else
        PYTHON=python3
    fi
fi
threads=${THREADS:-2}
device=${DEVICE:-auto}
export TOKENIZERS_PARALLELISM=false

slots=()
if [[ -n "${GPUS:-}" ]]; then
    if [[ ! "$GPUS" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ || "$device" == cpu ]]; then
        echo "GPUS expects comma-separated physical GPU indices and a CUDA device" >&2
        exit 2
    fi
    IFS=, read -r -a slots <<< "$GPUS"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    device=cuda
fi

variants=$("$PYTHON" -c 'from sidrec.common import NEIGHBOR_SWEEP; print("\n".join(NEIGHBOR_SWEEP))')
mapfile -t presets <<< "$variants"
flock data/.prepare.lock "$PYTHON" -m sidrec.prepare --neighbor-sweep --threads "$threads"

run_job() {
    local preset=$1 seed=$2 log_dir="runs/$1/neighbors/$2"
    mkdir -p "$log_dir"
    "$PYTHON" -m sidrec.train "$preset" neighbors "$seed" --device "$device" --threads "$threads" \
        2>&1 | tee -a "$log_dir/train.log"
}

job_presets=()
job_seeds=()
for preset in "${presets[@]}"; do
    for seed in 17 42 73; do
        job_presets+=("$preset")
        job_seeds+=("$seed")
    done
done

if (( ${#slots[@]} )); then
    pids=()
    for ((worker = 0; worker < ${#slots[@]} && worker < ${#job_presets[@]}; worker++)); do
        (
            export CUDA_VISIBLE_DEVICES="${slots[worker]}"
            for ((job = worker; job < ${#job_presets[@]}; job += ${#slots[@]})); do
                run_job "${job_presets[job]}" "${job_seeds[job]}"
            done
        ) &
        pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=$?
    done
    (( status == 0 )) || exit "$status"
    export CUDA_VISIBLE_DEVICES="${slots[0]}"
else
    for job in "${!job_presets[@]}"; do
        run_job "${job_presets[job]}" "${job_seeds[job]}"
    done
fi

"$PYTHON" -m sidrec.neighbor_sweep --device "$device" --threads "$threads"
