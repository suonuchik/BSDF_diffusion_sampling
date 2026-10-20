#!/bin/bash
# Render the NN (diffusion) yarn with and without the fixes (conda env bsdf35).
# Usage: run_nn_variants.sh <spp> <variant>...
#   variant: none | base | chart | both | all (both + MIS) | final (all + eval fix) | analytic | analytic_origeval
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bsdf35
cd "$(dirname "$0")" || exit 1
spp=$1; shift
for v in "$@"; do
  case $v in
    none)  flags="" ;;
    base)  flags="--fix_base" ;;
    chart) flags="--fix_chart" ;;
    both)  flags="--fix_base --fix_chart" ;;
    all)   flags="--fix_base --fix_chart --fix_mis" ;;
    analytic) flags="--analytic" ;;
    analytic_origeval) flags="--analytic --original_eval" ;;
    final) flags="--fix_base --fix_chart --fix_mis --fix_eval" ;;
    *) echo "unknown variant $v"; continue ;;
  esac
  start=$(date +%s)
  python render_yarn26_fibers_nn_patched.py --spp "$spp" --seed 0 $flags 2>&1 | grep -E 'Saved|Error|error|Traceback'
  echo "variant=$v spp=$spp elapsed $(( $(date +%s) - start )) s"
done
