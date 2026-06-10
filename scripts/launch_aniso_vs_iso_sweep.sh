#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J aniso_vs_iso
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=R-%x-%j.out
#
# Static alpha_dot relative error sweep: K=1 isotropic vs K in {2,3,4}
# per-cluster anisotropic-PCA, n_per_cluster in {25, 50, 100, 200, 400, 800}.
# 24 cells total; --workers 24 fits the entire wave in one shot.
#
# Memory: 24 workers x ~4 GB = 96 GB headroom. The K=4 n=800 cell
# (3200 features, full STLSQ with Tikhonov) is the peak; smaller cells
# finish quickly.
#
# Usage (cluster): sbatch scripts/launch_aniso_vs_iso_sweep.sh
# Usage (local):   bash   scripts/launch_aniso_vs_iso_sweep.sh

set -euo pipefail

echo "================================================="
echo "RBF_ROM aniso-vs-iso sweep"
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

# Single-threaded BLAS: parallelism is across cells, each cell does a
# sequential STLSQ + small linear algebra and benefits more from worker
# count than from threaded BLAS.
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
python -u scripts/run_aniso_vs_iso_sweep.py \
    --dataset LOR63 \
    --K-grid 1 2 3 4 \
    --n-grid 25 50 100 200 400 800 \
    --workers "$WORKERS"

# -- Done ---------------------------------------------------------------------
echo ""
echo "=========================================="
echo "JOB COMPLETED"
echo "=========================================="
echo "End time: $(date)"
