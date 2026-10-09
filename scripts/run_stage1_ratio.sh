#!/usr/bin/env bash
set -euo pipefail

resume_args=()
if [[ "${1:-}" == "--resume" ]]; then
    resume_args=(--skip-completed)
    shift
fi

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 [--resume] {50:50|60:40|70:30|80:20|90:10}" >&2
    exit 2
fi

split="$1"
case "$split" in
    50:50|60:40|70:30|80:20|90:10) ;;
    *)
        echo "Unsupported EA/FILL split: $split" >&2
        exit 2
        ;;
esac

project_root="${PROJECT_ROOT:-/workspace/adaptive-pareto-set-model}"
python_bin="${PYTHON_BIN:-/workspace/AF-model/.venv/bin/python}"
platemo_root="${PLATEMO_ROOT:-/workspace/PlatEMO/PlatEMO}"
tag="${split/:/_}"
output_root="${OUTPUT_DIR:-$project_root/results/stage1/ratio_$tag}"

cd "$project_root"
mkdir -p "$output_root"

"$python_bin" -u run_gd_psl.py \
    --algorithms NSGAII NSGAIII MOEAD \
    --problems re21 re24 \
    --population-size 100 \
    --max-fe 100000 \
    --candidate-pool-size 100000 \
    --ea-fill-split "$split" \
    --runs 20 \
    --seed 21 \
    --device cuda \
    "${resume_args[@]}" \
    --platemo-root "$platemo_root" \
    --output-dir "$output_root"

"$python_bin" -u run_gd_psl.py \
    --algorithms NSGAII NSGAIII MOEAD \
    --problems dtlz2 dtlz7 re31 re32 re34 re35 re37 \
    --n-objectives 3 \
    --population-size 91 \
    --max-fe 91000 \
    --candidate-pool-size 100000 \
    --ea-fill-split "$split" \
    --runs 20 \
    --seed 21 \
    --device cuda \
    "${resume_args[@]}" \
    --platemo-root "$platemo_root" \
    --output-dir "$output_root"
