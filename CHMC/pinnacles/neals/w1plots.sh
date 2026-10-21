#!/bin/bash
#SBATCH --job-name=neals_w1_plots
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=0-00:30:00
#SBATCH --output=neals_w1_plots_%j.out


cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

echo "w1_plots"
python -u /home/johngallagher/data/HMC-Research/CHMC/scripts/neals/plots.py