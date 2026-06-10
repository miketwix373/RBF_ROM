#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/CHORD2
#SBATCH -J l63_step2e
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --output=R-%x-%j.out
#
# L63 step 2e: lambda_rbf x sigma_A-pinning ablation for clustered RBF SINDy.
#   40 tasks = 4 K x 5 lambda x 2 sigma_modes at N_tot=1280.
#   Each task is a from-scratch clustered fit + inline diagnostics.
#   40 workers, 4 GB each. Small-lambda K=1 STLSQ on a 49997x1280 design
#   needs ~3.7 GB (OOM'd 8-worker @ 16 GB run); 4 GB/worker is comfortable.
#
# Usage (cluster): sbatch results/LOR63/codes/launch_step2e.sh
# Usage (local):   bash   results/LOR63/codes/launch_step2e.sh
# Extra args forwarded verbatim to step2e_ablate.py.

set -euo pipefail

EXTRA_ARGS=("$@")

echo "================================================="
echo "CHORD2 L63 step 2e -- lambda x sigma_A ablation"
echo "================================================="
echo "Start time: $(date)"
echo "Args:       ${EXTRA_ARGS[*]+${EXTRA_ARGS[*]}}"
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

# Single-threaded BLAS: parallelism is over (K, lambda, sigma_mode) tasks.
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
CMD=(python -u results/LOR63/codes/step2e_ablate.py "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}")

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
