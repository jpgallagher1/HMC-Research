#!/bin/bash
#SBATCH --job-name=neals_det
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=0-1:00:00
#SBATCH --output=neals_det_%j.out

# Neal's funnel with the log|det J| correction in the Metropolis ratio (config.logdet = True):
# a single setting, tau = 0.125, T = 3, d = 10 (set in timed_blocks_funnel_det.py).
# Runs LF, AA, Newt, FPI for every run in 200-sample blocks to 10,000 saved incrementally;
# LF and FPI then continue until their cumulative wall time matches AA's.
# Every block also saves the per-proposal log|det J| as `logdets` (zeros for leapfrog).
# Keys are recovered from ${KEYS_FROM} (its metadata grid), so every chain starts where the
# uncorrected chain did; HMC-LF reproduces it exactly. Resubmitting resumes from the last saved block.

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

SCRIPT_DIR=/home/johngallagher/data/HMC-Research/CHMC/scripts/NCC_SIAM_26
OUTPUT=/scratch/johngallagher/neals_funnel_det_samekeys_tau0.125_T3
KEYS_FROM=/scratch/johngallagher/neals_funnel_blocks   # uncorrected run; same keys and starting points

echo "Running Neal's funnel determinant run, tau=0.125, T=3"
python -u ${SCRIPT_DIR}/timed_blocks_funnel_det.py --output ${OUTPUT} --keys_from ${KEYS_FROM}
