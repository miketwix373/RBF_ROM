#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/RBF_ROM
#SBATCH -J aniso_vs_iso_lor96
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=R-%x-%j.out
#
# LOR96 vlachas_F8: K=1 isotropic vs K=10 anisotropic-PCA at matched
# total RBF count >= 6000. Asymmetric n_per_cluster grids via --cells.
#
# Matched totals:
#   total = 6000:  K=1 n=6000      vs  K=10 n=600
#   total = 10000: K=1 n=10000     vs  K=10 n=1000
#   total = 15000: K=1 n=15000     vs  K=10 n=1500
#
# Six cells, well under 24 workers. Memory budget: K=1 n=15000 Phi is
# ~20000 x 15000 float64 = 2.4 GB; with Tikhonov augmentation and copies
# allow ~25 GB per worker -> 192 GB headroom for the slowest cell + 5
# concurrent smaller cells.
#
# Wall time: K=1 cells dominate because kappa(Phi) blows up at large n
# (the L63 finding); K=1 n=15000 may need 60-90 min single-threaded.

set -euo pipefail

echo "================================================="
echo "RBF_ROM aniso-vs-iso (LOR96 vlachas_F8, large budget)"
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

python -u scripts/run_aniso_vs_iso_sweep.py \
    --dataset LOR96_vlachas_F8 \
    --stride 10 \
    --cells 1:6000 1:10000 1:15000 10:600 10:1000 10:1500 \
    --workers "$WORKERS"

echo ""
echo "=========================================="
echo "JOB COMPLETED"
echo "=========================================="
echo "End time: $(date)"
