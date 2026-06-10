#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/CHORD2
#SBATCH -J l63_step1_nrbf
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=R-%x-%j.out
#
# L63 RBF-only step 1: K=1, isotropic, n_rbf in
#   {5, 10, 20, 50, 100, 200, 400, 800, 1600, 3200, 6400}.
# Each cell fits one RBF-only model (seed=1, lambda_rbf=1e-3) and
# integrates 8 ICs RK4 to T=50 s. Outputs T_ph vs n_rbf, T_ph vs
# static-fit residual (the hypothesis-test plot), W1 vs n_rbf, and
# the stability-floor count. Parallel over the n_rbf grid; the upper
# end (3200, 6400) is the expensive piece (M=~50k x n_rbf design
# matrix, single-thread BLAS per worker).
#
# Usage (cluster):
#   sbatch results/LOR63/launch_step1_nrbf_sweep.sh
#   sbatch results/LOR63/launch_step1_nrbf_sweep.sh --workers 6
#
# Usage (local):
#   bash results/LOR63/launch_step1_nrbf_sweep.sh --workers 4
#
# Extra args are forwarded verbatim to step1_nrbf_sweep.py. The
# launcher injects --workers 8 unless the caller already set it
# (peak memory ~ 8 * 2.5 GB at n_rbf=6400 fits under 48 GB).

set -euo pipefail

EXTRA_ARGS=("$@")

HAS_WORKERS=false
for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
    if [ "$a" = "--workers" ]; then HAS_WORKERS=true; break; fi
done
if ! $HAS_WORKERS; then
    EXTRA_ARGS+=(--workers 8)
fi

echo "============================================="
echo "CHORD2 L63 step 1 -- K=1 isotropic n_rbf sweep"
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

# Pin BLAS to 1 thread per worker; the parallelism is the n_rbf grid,
# not parallel matmuls (per-cell matmuls saturate one core anyway).
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

# -- Run ----------------------------------------------------------------------
CMD=(python -u results/LOR63/codes/step1_nrbf_sweep.py "${EXTRA_ARGS[@]}")

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
