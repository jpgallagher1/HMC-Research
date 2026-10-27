#!/bin/bash
#SBATCH --job-name=pchi_plots
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --output=pchi_plots_%j.out

# Crossover plot for the folder written by pchi_blocks.sh: p=6, d=1200, tau=0.072, T=4,
# W1 over all retained states against time, log-log. Writes ${OUTPUT}/w1_metrics.npz and
# ${OUTPUT}/plot/w1_retained/tau_0_072/T_4_0/pchi_p6_d1200_w1_retained_time_loglog.png
# Add --ppt 1 for the 16 pt slide versions (plot_ppt/).

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

SCRIPT_DIR=/home/johngallagher/data/HMC-Research/CHMC/scripts/NCC_SIAM_26
P=6
D=1200
TAU=0.072
T=4.0
OUTPUT=/scratch/johngallagher/pchi_p${P}_d${D}_tau${TAU}_blocks

python -u ${SCRIPT_DIR}/timed_blocks_pchi_plots.py --p ${P} --d ${D} --tau ${TAU} --T ${T} \
    --metric w1_retained --xaxis time --scale loglog --output ${OUTPUT}
