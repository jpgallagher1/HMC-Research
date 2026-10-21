#!/bin/bash
#SBATCH --job-name=neals_scatter
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=0-00:10:00
#SBATCH --output=neals_scatter_%j.out

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

T_IDX=0        # Ts = linspace(1, 5, 5), so index 0 is T = 1

for TAU_IDX in 0 1 2 3 4; do
    echo "tau_idx $TAU_IDX, T_idx $T_IDX"
    python -u //home/johngallagher/data/HMC-Research/CHMC/scripts/neals/funnel_scatter.py "$TAU_IDX" "$T_IDX"
done