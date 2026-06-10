#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/CHORD2
#SBATCH -J l63_step2b
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=R-%x-%j.out
#
# L63 step 2b: Lyapunov spectrum + PDF survival for clustered RBF ROM.
#   K in {1,2,3,4} at N_tot=1280 (Stage A noise floor).
#   - 4 Lyapunov tasks: T = 1e4 Lyapunov times, tau_renorm = 0.5 LT.
#     Each is the bottleneck (~2.2 M tangent RK4 steps, ~15-20 min).
#   - 96 PDF tasks (4 K x 8 dirs x 3 scales): T = 200 LTs, ~few minutes.
#   8 workers; the 4 Lyapunov tasks pin 4 cores, PDFs fill the rest.
#
# Usage (cluster): sbatch results/LOR63/codes/launch_step2b.sh
# Usage (local):   bash   results/LOR63/codes/launch_step2b.sh
# Extra args are forwarded verbatim to step2b_stability.py.

set -euo pipefail

EXTRA_ARGS=("$@")

echo "================================================="
echo "CHORD2 L63 step 2b -- Lyapunov + PDF survival"
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

# Single-threaded BLAS: parallelism is over (K, IC) tasks, and each task
# is a sequential RK4 loop dominated by small RBF evals.
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
CMD=(python -u results/LOR63/codes/step2b_stability.py "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}")

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
