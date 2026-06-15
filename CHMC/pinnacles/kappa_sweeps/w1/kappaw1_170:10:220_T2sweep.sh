#!/bin/bash
#SBATCH --job-name=w1_170:10:220_T2sweep
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=0-01:00:00
#SBATCH --export=ALL

python -u /home/johngallagher/data/HMC-Research/CHMC/scripts/sweepw1_170:10:220_T.py 2 > ${SLURM_JOB_NAME}.stdout 2>&1