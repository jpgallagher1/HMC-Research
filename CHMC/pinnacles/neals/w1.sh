#!/bin/bash
#SBATCH --job-name=neals_w1
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=0-6:00:00
#SBATCH --array=0-5
#SBATCH --output=neals_w1_%A_%a.out

METHODS=(LF FPI Newton AA_m=4 AA_m=3 AA_m=2)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

echo "array task $SLURM_ARRAY_TASK_ID: $METHOD"
python -u /home/johngallagher/data/HMC-Research/CHMC/scripts/neals/w1metrics.py "$METHOD"