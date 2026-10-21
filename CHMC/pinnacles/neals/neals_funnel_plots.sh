#!/bin/bash
#SBATCH --job-name=neals_plots
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=0-4:00:00
#SBATCH --output=neals_plots_%j.out

# Sliced Wasserstein plots for the Neal's funnel timed blocks.
# Run after neals_funnel_blocks.sh has finished, e.g.
#   JOB=$(sbatch --parsable CHMC/pinnacles/neals/neals_funnel_blocks.sh)
#   sbatch --dependency=afterok:${JOB} CHMC/pinnacles/neals/neals_funnel_plots.sh
# Plots go to ${OUTPUT}/plot/sw/tau_*/T_*/ (or plot_ppt/ with --ppt True);
# the metric arrays are saved to ${OUTPUT}/sw_metrics.npz.

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

SCRIPT_DIR=/home/johngallagher/data/HMC-Research/CHMC/scripts/NCC_SIAM_26
OUTPUT=/scratch/johngallagher/neals_funnel_blocks

python -u ${SCRIPT_DIR}/timed_blocks_funnel_plots.py --output ${OUTPUT} --ppt 1
