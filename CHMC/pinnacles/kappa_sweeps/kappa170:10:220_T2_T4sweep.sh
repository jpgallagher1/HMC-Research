#!/bin/bash
#SBATCH --job-name=170:10:220_Tsweep
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --export=ALL

python -u /home/johngallagher/data/HMC-Research/CHMC/scripts/sweep170:10:220_T.py 2 > ${SLURM_JOB_NAME}_T2.stdout 2>&1 &
python -u /home/johngallagher/data/HMC-Research/CHMC/scripts/sweep170:10:220_T.py 4 > ${SLURM_JOB_NAME}_T4.stdout 2>&1 &

wait

