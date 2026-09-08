#!/bin/bash
#SBATCH --job-name=gmm_Newton_subregion
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=0-2:00:00
#SBATCH --output=gmm_Newton_subregion.out

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research
python -u /home/johngallagher/data/HMC-Research/CHMC/scripts/gmm/chmc_gmm_Newton.py