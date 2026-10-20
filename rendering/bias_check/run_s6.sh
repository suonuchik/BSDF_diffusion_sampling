#!/bin/bash
# Stage 6 sweep over incident directions (conda env bsdf35).  Usage: run_s6.sh [n] [--no_fix]
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bsdf35
cd "$(dirname "$0")" || exit 1
start=$(date +%s)
python s6_sweep_wi.py --n "${1:-131072}" "${@:2}" 2>&1 | grep -v -E 'Warning|run_backward|optix|OptiX'
echo "elapsed $(( $(date +%s) - start )) s"
