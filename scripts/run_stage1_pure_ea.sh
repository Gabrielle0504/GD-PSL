#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {NSGAII|NSGAIII|MOEAD}" >&2
    exit 2
fi

algorithm="${1^^}"
case "$algorithm" in
    NSGAII|NSGAIII|MOEAD) ;;
    *)
        echo "Unsupported algorithm: $1" >&2
        exit 2
        ;;
esac

project_root="${PROJECT_ROOT:-/workspace/adaptive-pareto-set-model}"
python_bin="${PYTHON_BIN:-/workspace/AF-model/.venv/bin/python}"
platemo_root="${PLATEMO_ROOT:-/workspace/PlatEMO/PlatEMO}"
output_dir="${OUTPUT_DIR:-$project_root/results/stage1/pure_ea}"

cd "$project_root"
mkdir -p "$output_dir"

common=(
    --algorithms "$algorithm"
    --runs 20
    --seed 21
    --ea-only
    --device cuda
    --platemo-root "$platemo_root"
    --output-dir "$output_dir"
    --skip-completed
    --no-plot
)

"$python_bin" run_fill_then_judge.py \
    "${common[@]}" \
    --problems re21 re24 \
    --population-size 100 \
    --max-fe 100000

"$python_bin" run_fill_then_judge.py \
    "${common[@]}" \
    --problems dtlz2 dtlz7 re31 re32 re34 re35 re37 \
    --n-objectives 3 \
    --population-size 91 \
    --max-fe 91000
