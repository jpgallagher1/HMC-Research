#!/bin/bash
#SBATCH --job-name=w1_172:2:188_T2_sweep
#SBATCH --output=%x.stdout
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=0-02:00:00
#SBATCH --export=ALL

python -u /home/johngallagher/data/HMC-Research/CHMC/scripts/sweepw1_9κ.py 2 172 188 9 
