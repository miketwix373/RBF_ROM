#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/CHORD2
#SBATCH -J l63_step2f
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
#SBATCH --time=02:30:00
#SBATCH --output=R-%x-%j.out
#
# L63 step 2f: Stage B (Lyapunov + PDF survival) on global-sigma_A refit.
#
# Two stages, run sequentially in one job:
#   (1) step2f_refit_globalsigma.py
#       Refits K in {1,2,3,4} at N_tot=1280, sigma_A pinned to the global
#       training std. Saves cells in step2a's schema so step2b's loader
#       picks them up transparently. Fast (~1 min on 4 workers).
#
#   (2) step2b_stability.py --models-dir <step2f models> --out-dir <step2f stab>
#       Lyapunov spectrum + PDF survival on F={1,2,3} off-attractor ICs.
#       Heavy: 4 Lyapunov tasks (T=1e4 LTs, ~15-20 min each on one core)
#       + 96 PDF tasks (4 K x 8 dirs x 3 F, ~few min each). 40 workers fit
#       the entire wave (4 + 96 = 100 tasks) with room to spare.
#
# Memory: 40 workers x 4 GB = 160 GB. Matches the 4 GB/worker budget that
# fixed the OOM in step 2e (small-lambda K=1 STLSQ peaks at ~3.7 GB).
#
# Usage (cluster): sbatch results/LOR63/codes/launch_step2f.sh
# Usage (local):   bash   results/LOR63/codes/launch_step2f.sh

set -euo pipefail

echo "================================================="
echo "CHORD2 L63 step 2f -- Stage B on global-sigma_A"
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

# Single-threaded BLAS: parallelism is over tasks; each task is a
# sequential RK4 / STLSQ loop dominated by small evals.
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

# -- Paths --------------------------------------------------------------------
STEP2F_DIR="results/LOR63/step2f_globalsigma"
MODELS_DIR="$STEP2F_DIR/models"
STAB_DIR="$STEP2F_DIR/stability"
mkdir -p "$STEP2F_DIR" "$STAB_DIR"

# -- (1) Refit cells with sigma_A pinned to global ----------------------------
echo "=========================================="
echo "STAGE 1/2: refit cells (global sigma_A)"
echo "=========================================="
python -u results/LOR63/codes/step2f_refit_globalsigma.py \
    --workers 4 \
    --out-dir "$STEP2F_DIR"

# -- (2) Stage B: Lyapunov + PDF survival on the refit cells ------------------
echo ""
echo "=========================================="
echo "STAGE 2/2: Lyapunov + PDF survival"
echo "=========================================="
python -u results/LOR63/codes/step2b_stability.py \
    --workers 40 \
    --F-list 1 2 3 \
    --models-dir "$MODELS_DIR" \
    --out-dir "$STAB_DIR"

# -- Done ---------------------------------------------------------------------
echo ""
echo "=========================================="
echo "JOB COMPLETED"
echo "=========================================="
echo "End time: $(date)"
