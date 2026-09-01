#!/bin/bash
#SBATCH --job-name=gmm_aa-2
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=0-6:00:00

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research
python -u /home/johngallagher/data/HMC-Research/CHMC/scripts/gmm/chmc_gmm_AA.py 2