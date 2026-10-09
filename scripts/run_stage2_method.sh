#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {EA_NSGAII|EA_NSGAIII|EA_MOEAD|GD-PSL_NSGAII|GD-PSL_NSGAIII|GD-PSL_MOEAD|PangMOEAD}" >&2
    exit 2
fi

method="$1"
project_root="${PROJECT_ROOT:-/workspace/adaptive-pareto-set-model}"
python_bin="${PYTHON_BIN:-/workspace/AF-model/.venv/bin/python}"
platemo_root="${PLATEMO_ROOT:-/workspace/PlatEMO/PlatEMO}"
output_root="${OUTPUT_DIR:-$project_root/results/stage2}"

cd "$project_root"
mkdir -p "$output_root"

run_gd_psl_or_ea() {
    local algorithm="$1"
    shift
    local mode_args=("$@")
    local common=(
        --algorithms "$algorithm"
        --runs 20
        --seed 101
        --device cuda
        --platemo-root "$platemo_root"
        --output-dir "$output_root"
        --skip-completed
        --no-plot
        "${mode_args[@]}"
    )

    "$python_bin" -u run_fill_then_judge.py \
        "${common[@]}" \
        --problems re21 re24 \
        --population-size 100 \
        --max-fe 100000

    "$python_bin" -u run_fill_then_judge.py \
        "${common[@]}" \
        --problems dtlz2 dtlz7 re31 re32 re34 re35 re37 \
        --n-objectives 3 \
        --population-size 91 \
        --max-fe 91000
}

run_pang() {
    local common=(
        --algorithms PangMOEAD
        --runs 20
        --seed 101
        --platemo-root "$platemo_root"
        --output-dir "$output_root"
        --skip-completed
    )

    "$python_bin" -u run_pang_archive_baseline.py \
        "${common[@]}" \
        --problems re21 re24 \
        --population-size 100 \
        --max-fe 100000

    "$python_bin" -u run_pang_archive_baseline.py \
        "${common[@]}" \
        --problems dtlz2 dtlz7 re31 re32 re34 re35 re37 \
        --n-objectives 3 \
        --population-size 91 \
        --max-fe 91000
}

case "$method" in
    EA_NSGAII) run_gd_psl_or_ea NSGAII --ea-only ;;
    EA_NSGAIII) run_gd_psl_or_ea NSGAIII --ea-only ;;
    EA_MOEAD) run_gd_psl_or_ea MOEAD --ea-only ;;
    GD-PSL_NSGAII) run_gd_psl_or_ea NSGAII --ea-fill-split 90:10 ;;
    GD-PSL_NSGAIII) run_gd_psl_or_ea NSGAIII --ea-fill-split 90:10 ;;
    GD-PSL_MOEAD) run_gd_psl_or_ea MOEAD --ea-fill-split 90:10 ;;
    PangMOEAD) run_pang ;;
    *)
        echo "Unsupported Stage 2 method: $method" >&2
        exit 2
        ;;
esac
