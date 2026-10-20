#!/bin/bash
# Stage 2 single-bounce test (conda env bsdf35).  Usage: run_s2.sh [n] [--fix_base] [--fix_chart] [--theta_i ...]
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bsdf35
cd "$(dirname "$0")" || exit 1
start=$(date +%s)
python s2_single_bounce.py --n "${1:-524288}" --batch 131072 "${@:2}" 2>&1 | grep -v -E 'Warning|run_backward|optix|OptiX'
echo "elapsed $(( $(date +%s) - start )) s"
