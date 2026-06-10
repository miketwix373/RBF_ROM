#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J aniso_vs_iso_diag
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=R-%x-%j.out
#
# Diagnostics on the aniso-vs-iso sweep fits:
#   - Dead-zone shell coverage at d-grid {0.5, 1, 2, 3} and tau in {0.1, 0.3}
#   - Forward RK4 integration from 8 shared ICs (T = 50 s)
#   - Eq.21 T_ph, finite-at-T, marginal PDFs, Wasserstein-1, kappa(Phi)
#
# 24 cells; --workers 24 fits the whole wave. Each cell dominated by
# RK4 of 5000 steps x 8 ICs through up to 3200 anisotropic features
# (K=4 n=800 cell). Estimated wall time 20-40 min.
#
# Usage (cluster): sbatch scripts/launch_aniso_vs_iso_diagnostics.sh
# Usage (local):   bash   scripts/launch_aniso_vs_iso_diagnostics.sh

set -euo pipefail

echo "================================================="
echo "RBF_ROM aniso-vs-iso diagnostics"
echo "================================================="
echo "Start time: $(date)"
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

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# -- Job info -----------------------------------------------------------------
echo "=========================================="
echo "SLURM JOB"
echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID:-(local)}"
echo "Node:          ${SLURM_JOB_NODELIST:-(local)}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-(local)}"
echo "Memory:        ${SLURM_MEM_PER_NODE:-(local)} MB"
echo ""

WORKERS="${SLURM_CPUS_PER_TASK:-24}"

# -- Run ----------------------------------------------------------------------
python -u scripts/run_aniso_vs_iso_diagnostics.py \
    --dataset LOR63 \
    --workers "$WORKERS"

# -- Done ---------------------------------------------------------------------
echo ""
echo "=========================================="
echo "JOB COMPLETED"
echo "=========================================="
echo "End time: $(date)"
