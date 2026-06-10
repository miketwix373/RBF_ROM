#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/CHORD2
#SBATCH -J l63_step1b_pdf
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=R-%x-%j.out
#
# L63 step 1b: marginal-PDF grid for 6 curated n_rbf values.
# Fits each + integrates 8 ICs RK4 to T=50 s, pools the tail
# t in [10, 50] s, plots (n_rbf rows) x (x, y, z cols) histograms
# overlaid on the FOM truth PDF.
#
# Usage:
#   sbatch results/LOR63/codes/launch_step1b_pdf_grid.sh
#   sbatch results/LOR63/codes/launch_step1b_pdf_grid.sh --n-rbf 5 100 800

set -euo pipefail

EXTRA_ARGS=("$@")

HAS_WORKERS=false
for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
    if [ "$a" = "--workers" ]; then HAS_WORKERS=true; break; fi
done
if ! $HAS_WORKERS; then
    EXTRA_ARGS+=(--workers 6)
fi

echo "============================================="
echo "CHORD2 L63 step 1b -- PDF grid (curated n_rbf)"
echo "============================================="
echo "Start time: $(date)"
echo "Args:       ${EXTRA_ARGS[*]}"
echo ""

if command -v flight >/dev/null 2>&1; then
    flight env activate gridware
fi

__conda_setup="$('/mnt/scratch/users/sbrw610/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
fi
unset __conda_setup

conda activate /mnt/scratch/users/sbrw610/anaconda3/envs/cfd_new

export MKL_THREADING_LAYER=GNU
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="/opt/apps/flight/env/conda+jupyter/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "=========================================="
echo "SLURM JOB"
echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID:-(local)}"
echo "Node:          ${SLURM_JOB_NODELIST:-(local)}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-(local)}"
echo "Memory:        ${SLURM_MEM_PER_NODE:-(local)} MB"
echo ""

CMD=(python -u results/LOR63/codes/step1b_pdf_grid.py "${EXTRA_ARGS[@]}")

echo "=========================================="
echo "RUNNING ${CMD[*]}"
echo "=========================================="
"${CMD[@]}"

echo ""
echo "=========================================="
echo "JOB COMPLETED"
echo "=========================================="
echo "End time: $(date)"
