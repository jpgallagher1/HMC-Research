#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition test
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --array=0-8               # 0 to (numτs × mcmc_iter_count − 1); update upper bound if params change
#SBATCH --output=chains_%A_%a.stdout
#SBATCH --job-name=hmc_chains
##SBATCH --mail-user=UCMercedNetID@ucmerced.edu
##SBATCH --mail-type=ALL
#SBATCH --export=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../scripts/results/chains"

# Limit JAX/XLA thread count to match ntasks-per-node
export XLA_FLAGS="--xla_cpu_multi_thread_eigen_intra_op_parallelism=${SLURM_NTASKS_PER_NODE}"
export OMP_NUM_THREADS=${SLURM_NTASKS_PER_NODE}

python "${SCRIPT_DIR}/../scripts/run_chains_task.py" \
    --task_id    "${SLURM_ARRAY_TASK_ID}" \
    --output_dir "${RESULTS_DIR}"
