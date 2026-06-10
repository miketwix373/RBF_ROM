#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/CHORD2
#SBATCH -J l63_adot_cmp
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=R-%x-%j.out
#
# L63 alpha-dot prediction comparison: poly-only vs RBF-only vs truth.
# Fits a single poly-only baseline (~9 nonzero coefficients) and a single
# RBF-only cell at the upper end of the n_rbf sweep, then plots a 10s
# window of (x_dot, y_dot, z_dot) plus a residual panel.
#
# Usage (cluster):
#   sbatch results/LOR63/launch_plot_alpha_dot_compare.sh
#   sbatch results/LOR63/launch_plot_alpha_dot_compare.sh --t-start 350 --window 10
#
# Usage (local, no SLURM):
#   bash results/LOR63/launch_plot_alpha_dot_compare.sh
#
# Extra args are forwarded verbatim to
# `results/LOR63/plot_alpha_dot_compare.py`.

set -euo pipefail

EXTRA_ARGS=("$@")

echo "============================================="
echo "CHORD2 L63 alpha-dot poly vs RBF comparison"
echo "============================================="
echo "Start time: $(date)"
echo "Args:       ${EXTRA_ARGS[*]}"
echo ""

# -- Environment --------------------------------------------------------------
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

# Single fit per model + single 10s window plot; no parallelism needed but
# pin BLAS to a small thread budget anyway so the node is well-behaved
# under shared use.
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

# -- Job info -----------------------------------------------------------------
echo "=========================================="
echo "SLURM JOB"
echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID:-(local)}"
echo "Node:          ${SLURM_JOB_NODELIST:-(local)}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-(local)}"
echo "Memory:        ${SLURM_MEM_PER_NODE:-(local)} MB"
echo ""

# -- Run ----------------------------------------------------------------------
CMD=(python -u results/LOR63/plot_alpha_dot_compare.py "${EXTRA_ARGS[@]}")

echo "=========================================="
echo "RUNNING ${CMD[*]}"
echo "=========================================="
"${CMD[@]}"

# -- Done ---------------------------------------------------------------------
echo ""
echo "=========================================="
echo "JOB COMPLETED"
echo "=========================================="
echo "End time: $(date)"
