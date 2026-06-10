#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/CHORD2
#SBATCH -J phase0_l63_rob
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=R-%x-%j.out
#
# Phase 0 L63 negative-control robustness sweep (Layer 1 + Layer 2).
# CPU-only: the per-fit linear algebra is tiny (M=50000 x N_feat<=210)
# so GPU offers no speedup; the 4 x 5 x 5 = 100 independent fits are
# embarrassingly parallel over CPU cores instead.
#
# Usage (cluster):
#   sbatch scripts/launch_phase0_l63_robustness.sh
#   sbatch scripts/launch_phase0_l63_robustness.sh --n-rbf 50 100 --workers 8
#
# Usage (local, no SLURM):
#   bash scripts/launch_phase0_l63_robustness.sh --workers 8
#
# Extra args are forwarded verbatim to
# `scripts/run_phase0_l63_robustness.py`. The launcher injects
# `--workers $SLURM_CPUS_PER_TASK` unless the caller passed --workers.

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
echo "CHORD2 Phase 0 L63 robustness sweep"
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

# Pin BLAS/MKL to a single thread per worker so 16 workers x N MKL threads
# does not oversubscribe the 16 allocated cores. Each fit is tiny in the
# linear algebra; the gain is from parallel cells, not parallel matmuls.
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
CMD=(python -u -m scripts.run_phase0_l63_robustness "${EXTRA_ARGS[@]}")

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
