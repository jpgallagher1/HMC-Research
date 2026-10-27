#!/bin/bash
#SBATCH --job-name=pchi_blocks
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=0-4:00:00
#SBATCH --array=0-3
#SBATCH --output=pchi_blocks_%A_%a.out

# p-chi timed blocks: one array task per (p, d), each with its own output folder.
#   0: p=4, d=800,  tau 0.1
#   1: p=4, d=1000, tau 0.1
#   2: p=4, d=1200, tau 0.125 0.16
#   3: p=6, d=1200, tau 0.0565 0.0625 0.072
# T = 4, 10 runs, starts q0 ~ N(mode, 1/U''(mode)). Each task runs AA first to 10,000 samples
# in 200-sample blocks saved incrementally; LF and FPI then continue until their cumulative
# wall time matches AA's. Resubmitting resumes from the last saved block of each chain.
# Plotting is run separately: pchi_plots.sh

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

SCRIPT_DIR=/home/johngallagher/data/HMC-Research/CHMC/scripts/NCC_SIAM_26

PS=(4 4 4 6)
DS=(800 1000 1200 1200)
TAUS=("0.1" "0.1" "0.125 0.16" "0.0565 0.0625 0.072")

P=${PS[$SLURM_ARRAY_TASK_ID]}
D=${DS[$SLURM_ARRAY_TASK_ID]}
OUTPUT=/scratch/johngallagher/pchi_p${P}_d${D}_blocks

echo "Running p-chi timed blocks, p=${P}, d=${D}, taus: ${TAUS[$SLURM_ARRAY_TASK_ID]}"
python -u ${SCRIPT_DIR}/timed_blocks_pchi.py --p ${P} --d ${D} --taus ${TAUS[$SLURM_ARRAY_TASK_ID]} --output ${OUTPUT}
