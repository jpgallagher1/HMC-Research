#!/bin/bash
#SBATCH --job-name=gmm_w1_loglog_time
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=0-1:00:00

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research
python -u /home/johngallagher/data/HMC-Research/CHMC/scripts/gmm/w1_plotting/w1plots.py