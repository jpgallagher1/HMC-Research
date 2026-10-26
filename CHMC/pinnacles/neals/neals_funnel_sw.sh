#!/bin/bash
#SBATCH --job-name=neals_sw
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-6:00:00
#SBATCH --output=neals_sw_%j.out

# Sliced Wasserstein (10_000 reference samples, 1000 projections, p = 1) for the Neal's funnel
# timed blocks: compute only, no plots. Every chain is evaluated at every saved length, in
# parallel over chains using all the cores requested above (--cpus-per-task; set it to the
# cores per node you want to use).
# Each (method, tau, T, run) result is saved to ${OUTPUT}/sw_data/ after every length, so a
# job that runs out of time resumes where it stopped when resubmitted.
# Run after neals_funnel_blocks.sh has finished, and plot afterwards with neals_funnel_plots.sh:
#   JOB=$(sbatch --parsable CHMC/pinnacles/neals/neals_funnel_blocks.sh)
#   JOB=$(sbatch --parsable --dependency=afterok:${JOB} CHMC/pinnacles/neals/neals_funnel_sw.sh)
#   sbatch --dependency=afterok:${JOB} CHMC/pinnacles/neals/neals_funnel_plots.sh
# Rough cost: about 14 core-hours at 1000 projections, so about 14 h / cores of wall time;
# memory is up to ~3.5 GB per core on the longest chains.

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

SCRIPT_DIR=/home/johngallagher/data/HMC-Research/CHMC/scripts/NCC_SIAM_26
OUTPUT=/scratch/johngallagher/neals_funnel_blocks

python -u ${SCRIPT_DIR}/timed_blocks_funnel_sw.py --output ${OUTPUT} --workers ${SLURM_CPUS_PER_TASK}
