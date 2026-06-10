#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/CHORD2
#SBATCH -J l63_int_cmp
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=R-%x-%j.out
#
# L63 RBF-only integration experiment (rom-specialist plan, steps 1-5):
# short-window IC0, long-window 8-IC attractor, marginal PDFs + W1,
# vector-field residual vs nearest-centre distance, symmetry residual,
# RK4 step-halving sanity. Single poly-only fit + single RBF-only fit
# (n_rbf=1600 upper end of the n_rbf sweep, seed=1 the lowest-residual
# cell). 8 ICs * (T/dt + 1) RK4 steps + one dt/2 confirmation pass.
#
# Usage (cluster):
#   sbatch results/LOR63/launch_integrate_compare.sh
#   sbatch results/LOR63/launch_integrate_compare.sh --T 100 --n-ic 16
#
# Usage (local, no SLURM):
#   bash results/LOR63/launch_integrate_compare.sh
#
# Extra args are forwarded verbatim to
# `results/LOR63/integrate_compare.py`.

set -euo pipefail

EXTRA_ARGS=("$@")

echo "============================================="
echo "CHORD2 L63 integration experiment"
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

# Two fits (poly-only + RBF-only) + 8-IC RK4 + step-halving + cloud
# diagnostics. RHS at n_rbf=1600 is a (1, 1600) Gaussian + matvec; the
# inner loop is small so a modest BLAS thread budget suffices.
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
CMD=(python -u results/LOR63/integrate_compare.py "${EXTRA_ARGS[@]}")

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
