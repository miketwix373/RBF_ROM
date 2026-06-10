#!/bin/bash
#SBATCH -D /mnt/scratch/users/sbrw610/CHORD2
#SBATCH -J l63_step0_shell
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=R-%x-%j.out
#
# Step 0 of the RBF-only L63 integration plan: field-magnitude shell test.
# Fits a single poly-only baseline + a single RBF-only cell at the upper
# end of the n_rbf sweep, samples 10k FOM-cloud and +10%-shell points,
# reports ||f|| distributions and shell/cloud ratios. If the RBF ratio is
# < 0.1, orbital escape is provable from dictionary geometry alone and
# the full integration experiment can be short-circuited.
#
# Usage (cluster):
#   sbatch results/LOR63/launch_step0_field_shell.sh
#   sbatch results/LOR63/launch_step0_field_shell.sh --shell-scale 1.20
#
# Usage (local, no SLURM):
#   bash results/LOR63/launch_step0_field_shell.sh
#
# Extra args are forwarded verbatim to
# `results/LOR63/step0_field_shell.py`.

set -euo pipefail

EXTRA_ARGS=("$@")

echo "============================================="
echo "CHORD2 L63 step 0 -- field shell test"
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

# Single fit + 10k-sample shell evaluation; pin BLAS to a small thread
# budget so the node is well-behaved under shared use.
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

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
CMD=(python -u results/LOR63/step0_field_shell.py "${EXTRA_ARGS[@]}")

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
