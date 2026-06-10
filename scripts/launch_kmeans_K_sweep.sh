#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J kmeans_K_sweep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=R-%x-%j.out
#
# K-sweep for LOR96 vlachas_F8, K in [1, 20], stride 10.
# One K per worker; BIC + residence + V_k subspace-angle diagnostics.
#
# Usage (cluster): sbatch scripts/launch_kmeans_K_sweep.sh
# Usage (local):   bash   scripts/launch_kmeans_K_sweep.sh

set -euo pipefail

echo "================================================="
echo "RBF_ROM K-means K sweep (LOR96 vlachas_F8)"
echo "================================================="
echo "Start time: $(date)"
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

WORKERS="${SLURM_CPUS_PER_TASK:-24}"

python -u scripts/run_kmeans_K_sweep.py \
    --stats-path data/LOR96/results/vlachas_F8/stats.npz \
    --out-dir    results/LOR96_vlachas_F8/K_sweep \
    --K-min 1 \
    --K-max 20 \
    --stride 10 \
    --r-compare 5 \
    --seed 0 \
    --n-init 10 \
    --workers "$WORKERS"

echo ""
echo "=========================================="
echo "JOB COMPLETED"
echo "=========================================="
echo "End time: $(date)"
