#!/bin/bash
#SBATCH --job-name=pchi_plots
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --array=0-3
#SBATCH --output=pchi_plots_%A_%a.out

# Plots and W1 metrics for the folders written by pchi_blocks.sh, same array index -> (p, d).
# Writes ${OUTPUT}/w1_metrics.npz and ${OUTPUT}/plot/{w1,w1_retained}/tau_*/T_*/.
# Add --ppt 1 for the 16 pt slide versions (plot_ppt/).

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

SCRIPT_DIR=/home/johngallagher/data/HMC-Research/CHMC/scripts/NCC_SIAM_26

PS=(4 4 4 6)
DS=(800 1000 1200 1200)

P=${PS[$SLURM_ARRAY_TASK_ID]}
D=${DS[$SLURM_ARRAY_TASK_ID]}
OUTPUT=/scratch/johngallagher/pchi_p${P}_d${D}_blocks

python -u ${SCRIPT_DIR}/timed_blocks_pchi_plots.py --output ${OUTPUT}
