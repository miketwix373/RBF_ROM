#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/CHORD2
#SBATCH -J phase0_l63_rbf
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=R-%x-%j.out
#
# Phase 0 L63 RBF-only counterfactual sweep. Companion to the joint
# poly + RBF robustness sweep: drop the polynomial block entirely and
# measure how well an RBF dictionary alone can reconstruct alpha_dot
# as a function of n_rbf. n_rbf-axis x seeds independent fits, each
# CPU-bound, fanned out over the allocated cores via
# multiprocessing.Pool inside the driver.
#
# Usage (cluster):
#   sbatch scripts/launch_phase0_l63_rbf_only.sh
#   sbatch scripts/launch_phase0_l63_rbf_only.sh --n-rbf 50 100 200 --workers 8
#
# Usage (local, no SLURM):
#   bash scripts/launch_phase0_l63_rbf_only.sh --workers 8
#
# Extra args are forwarded verbatim to
# `scripts/run_phase0_l63_rbf_only.py`. The launcher injects
# `--workers $SLURM_CPUS_PER_TASK` unless the caller already passed
# --workers.

set -euo pipefail

EXTRA_ARGS=("$@")

# Default --workers to $SLURM_CPUS_PER_TASK unless the user overrode it.
HAS_WORKERS=false
for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
    if [ "$a" = "--workers" ]; then HAS_WORKERS=true; break; fi
done
if ! $HAS_WORKERS; then
    EXTRA_ARGS+=(--workers "${SLURM_CPUS_PER_TASK:-48}")
fi

echo "============================================="
echo "CHORD2 Phase 0 L63 RBF-only sweep"
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

# Pin BLAS/MKL to a single thread per worker so N workers x M MKL threads
# does not oversubscribe the allocated cores. Per-fit linalg is small
# (M_mid x n_rbf, n_rbf <= ~1600); the gain is from parallel (n_rbf, seed)
# cells, not parallel matmuls.
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
CMD=(python -u -m scripts.run_phase0_l63_rbf_only "${EXTRA_ARGS[@]}")

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
