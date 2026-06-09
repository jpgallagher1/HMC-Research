#!/bin/bash
#SBATCH --job-name=20k:10k:100k_T1_sweep
#SBATCH --output=%x.stdout
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=0-00:40:00
#SBATCH --export=ALL

python -u /home/johngallagher/data/HMC-Research/CHMC/scripts/sweep_9κ.py 1 20000 100000 9 
