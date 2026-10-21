#!/bin/bash
#SBATCH --job-name=neals_aa
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=0-3:00:00
#SBATCH --array=2-4
#SBATCH --output=neals_AA_%a.out

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

echo "Running AA_m=${SLURM_ARRAY_TASK_ID}"

python -u /home/johngallagher/data/HMC-Research/CHMC/scripts/neals/chmc_neals_AA.py ${SLURM_ARRAY_TASK_ID}