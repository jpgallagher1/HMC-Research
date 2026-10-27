#!/bin/bash
#SBATCH --job-name=neals_seg
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-1:00:00
#SBATCH --output=neals_seg_%j.out

# Segment diagnostic for the Neal's funnel det run: splits every saved chain into consecutive
# SEGMENT-sample pieces and reports y min/max, max|x|, acceptance and the sliced W_1 of each
# piece alone (200 projections, same 10_000-sample reference as neals_funnel_sw_det.sh).
# Reads only the block files, so it can run any time after (or during) neals_funnel_blocks_det.sh.
# Table goes to this job's .out file; CSV to ${OUTPUT}/segment_diagnostics.csv (overwritten).
# Serial; a few minutes for ~1M saved samples at SEGMENT=1000.
#   sbatch CHMC/pinnacles/neals/neals_funnel_segments_det.sh
#   sbatch --export=ALL,SEGMENT=500 CHMC/pinnacles/neals/neals_funnel_segments_det.sh

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

SCRIPT_DIR=/home/johngallagher/data/HMC-Research/CHMC/scripts/NCC_SIAM_26
OUTPUT=/scratch/johngallagher/neals_funnel_det_samekeys_tau0.125_T3
SEGMENT=${SEGMENT:-1000}

echo "Segment diagnostic: ${OUTPUT}, segment=${SEGMENT}"
python -u ${SCRIPT_DIR}/timed_blocks_funnel_segments_det.py --output ${OUTPUT} --segment ${SEGMENT}
