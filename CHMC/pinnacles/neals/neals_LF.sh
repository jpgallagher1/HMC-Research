#!/bin/bash
#SBATCH --job-name=neals_LF
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=0-3:00:00
#SBATCH --output=neals_LF.out

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research
python -u /home/johngallagher/data/HMC-Research/CHMC/scripts/neals/chmc_neals_LF.py