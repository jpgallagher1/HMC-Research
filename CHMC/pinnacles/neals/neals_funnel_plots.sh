#!/bin/bash
#SBATCH --job-name=neals_plots
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=0-1:00:00
#SBATCH --output=neals_plots_%j.out

# Plots only: reads ${OUTPUT}/sw_data written by neals_funnel_sw.sh and computes nothing.
# Plots go to ${OUTPUT}/plot/sw/tau_*/T_*/ (or plot_ppt/ with --ppt True).
# Add --ppt True to the python line for the 16 pt slide versions.

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

SCRIPT_DIR=/home/johngallagher/data/HMC-Research/CHMC/scripts/NCC_SIAM_26
OUTPUT=/scratch/johngallagher/neals_funnel_blocks

python -u ${SCRIPT_DIR}/timed_blocks_funnel_plots.py --output ${OUTPUT}
