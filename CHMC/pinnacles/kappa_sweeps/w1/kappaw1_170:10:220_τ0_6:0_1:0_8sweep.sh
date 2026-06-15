#!/bin/bash
#SBATCH --job-name=w1_170:10:220_τsweep
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-3
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --export=ALL

taus=(0.6 0.7 0.8)
tau=${taus[$((SLURM_ARRAY_TASK_ID -1))]}

python -u /home/johngallagher/data/HMC-Research/CHMC/scripts/sweepw1_170:10:220_τ.py $tau > ${SLURM_JOB_NAME}_τ${tau//./_}.stdout 2>&1

