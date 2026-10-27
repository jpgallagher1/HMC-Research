#!/bin/bash
#SBATCH --job-name=neals_blocks
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=0-4:00:00
#SBATCH --output=neals_blocks_%j.out

# Neal's funnel timed blocks with log|det J|: a single setting, tau = 0.125, T = 3, d = 10.
# Runs LF, AA, Newt, FPI for every run in 200-sample blocks to 10,000 saved incrementally;
# LF and FPI then continue until their cumulative wall time matches AA's.
# Keys are recovered from ${KEYS_FROM} (its metadata grid), so every chain starts where the
# uncorrected chain did; HMC-LF reproduces it exactly. Resubmitting resumes from the last saved block.
# Then neals_funnel_sw_det.sh and neals_funnel_plots_det.sh.

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

SCRIPT_DIR=/home/johngallagher/data/HMC-Research/CHMC/scripts/NCC_SIAM_26
OUTPUT=/scratch/johngallagher/neals_funnel_det_samekeys_tau0.125_T3
KEYS_FROM=/scratch/johngallagher/neals_funnel_blocks   # uncorrected run; same keys and starting points

echo "Running Neal's funnel timed blocks, tau=0.125, T=3"
python -u ${SCRIPT_DIR}/timed_blocks_funnel_det.py --output ${OUTPUT} --keys_from ${KEYS_FROM}
