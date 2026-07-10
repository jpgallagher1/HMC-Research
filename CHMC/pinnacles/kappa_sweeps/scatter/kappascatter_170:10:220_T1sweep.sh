#!/bin/bash
#SBATCH --job-name=scatter_170:10:220_T1sweep
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=0-01:00:00
#SBATCH --export=ALL

python -u CHMC/scripts/scatterchmc_170:10:220_T.py 1 > ${SLURM_JOB_NAME}.stdout 2>&1