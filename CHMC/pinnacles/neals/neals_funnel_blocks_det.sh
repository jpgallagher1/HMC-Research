#!/bin/bash
#SBATCH --job-name=neals_blocks
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=0-4:00:00
#SBATCH --array=1-3
#SBATCH --output=neals_blocks_%A_%a.out

# Neal's funnel timed blocks: one array task per final time T = 1..5.
# Each task runs LF, AA, Newt, FPI for every tau and run at its T, in 200-sample blocks
# to 10,000 saved incrementally; LF/AA/FPI then continue until their cumulative wall time
# matches Newt's. All tasks write to the same output folder (disjoint T_* subfolders).
# Resubmitting resumes from the last saved block of each chain.
# Plotting is run separately: timed_blocks_funnel_plots.py --output ${OUTPUT}

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

SCRIPT_DIR=/home/johngallagher/data/HMC-Research/CHMC/scripts/NCC_SIAM_26
OUTPUT=/scratch/johngallagher/neals_funnel_blocks_det

echo "Running Neal's funnel timed blocks, T=${SLURM_ARRAY_TASK_ID}"
python -u ${SCRIPT_DIR}/timed_blocks_funnel_det.py --output ${OUTPUT} --only_T ${SLURM_ARRAY_TASK_ID}
