#!/bin/bash
#SBATCH --job-name=diag_newton
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --time=0-0:30:00
#SBATCH --output=diag_newton_%j.out

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

for TAU_IDX in 0 1 2 3 4; do
    python -u /home/johngallagher/data/HMC-Research/CHMC/scripts/neals/diagnosis.py "$TAU_IDX"
done