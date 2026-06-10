#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/CHORD2
#SBATCH -J phase0_l63_backfit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=R-%x-%j.out
#
# Phase 0 L63 negative-control under the backfit architecture. Sweeps
# n_rbf in {20, 50, 100} with the locked flat-isotropic RBF recipe
# (lambda_poly=1.0, lambda_rbf=0.5, gamma=1.0, n_knn=5). Reports the
# four locked L63 acceptance conditions plus the backfit-invariant
# functional-norm restatement of cond1. Saves per-fit models under
# results/LOR63/phase0_backfit/models/.
#
# Usage:
#   sbatch scripts/launch_phase0_l63_backfit.sh
#   sbatch scripts/launch_phase0_l63_backfit.sh --n-rbf 20

set -euo pipefail

EXTRA_ARGS=("$@")

echo "============================================="
echo "CHORD2 Phase 0 L63 negative-control backfit"
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

export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

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
CMD=(python -u -m scripts.run_phase0_l63_backfit "${EXTRA_ARGS[@]}")

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
