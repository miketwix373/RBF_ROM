#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/CHORD2
#SBATCH -J l63_step1d_risk
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=R-%x-%j.out
#
# L63 step 1d: off-cloud risk-surface diagnostics from saved models.
# Pure post-processing (no re-fitting); loads .npz files from step 1c.
#
# Usage:
#   sbatch results/LOR63/codes/launch_step1d_risk_surface.sh
#   sbatch results/LOR63/codes/launch_step1d_risk_surface.sh --z-slice 0 27

set -euo pipefail

EXTRA_ARGS=("$@")

echo "============================================="
echo "CHORD2 L63 step 1d -- risk surface diagnostics"
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

# Allow BLAS to use multiple threads here — single process, no pool
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "=========================================="
echo "SLURM JOB"
echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID:-(local)}"
echo "Node:          ${SLURM_JOB_NODELIST:-(local)}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-(local)}"
echo "Memory:        ${SLURM_MEM_PER_NODE:-(local)} MB"
echo ""

CMD=(python -u results/LOR63/codes/step1d_risk_surface.py "${EXTRA_ARGS[@]}")

echo "=========================================="
echo "RUNNING ${CMD[*]}"
echo "=========================================="
"${CMD[@]}"

echo ""
echo "=========================================="
echo "JOB COMPLETED"
echo "=========================================="
echo "End time: $(date)"
