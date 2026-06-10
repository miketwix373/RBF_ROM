#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/CHORD2
#SBATCH -J l63_step2a
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=R-%x-%j.out
#
# L63 step 2a: clustered RBF-only alpha_dot error vs N_tot.
#   K in {1,2,3,4}, N_tot in {20,40,80,160,320,640,1280}.
#   Two allocation rules per K (count, variance) -- 56 fits total.
# Serial; the heaviest cell is K=1, N_tot=1280 (~10 s). Total under
# 5 min. SLURM wall is 30 min for headroom.
#
# Usage (cluster):  sbatch results/LOR63/codes/launch_step2a.sh
# Usage (local):    bash   results/LOR63/codes/launch_step2a.sh
# Extra args are forwarded verbatim to step2a_cluster_rbf_fit.py.

set -euo pipefail

EXTRA_ARGS=("$@")

echo "================================================="
echo "CHORD2 L63 step 2a -- clustered RBF-only alpha_dot"
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

# Single-threaded BLAS: this sweep is serial and the SVD inside each
# STLSQ fit is small (worst case ~1280 x 1280). One BLAS thread suffices.
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
CMD=(python -u results/LOR63/codes/step2a_cluster_rbf_fit.py "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}")

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
