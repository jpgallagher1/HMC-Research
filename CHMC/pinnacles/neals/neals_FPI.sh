#!/bin/bash
#SBATCH --job-name=neals_FPI
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=0-2:00:00
#SBATCH --output=neals_FPI.out

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research
python -u /home/johngallagher/data/HMC-Research/CHMC/scripts/neals/chmc_neals_FPI.py