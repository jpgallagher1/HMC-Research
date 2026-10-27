#!/bin/bash
#SBATCH --job-name=pchi_blocks
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=0-1:00:00
#SBATCH --output=pchi_blocks_%j.out

# p-chi timed blocks at the LF/CHMC crossover: p=6, d=1200, tau=0.072, T=4 (tau*omega = 1.87,
# omega = sqrt(p)(d-1)^{(p-2)/(2p)}), 10 runs, starts q0 ~ N(mode, 1/U''(mode)).
# AA runs first to 10,000 samples in 200-sample blocks saved incrementally; LF and FPI then
# continue until their cumulative wall time matches AA's. Resubmitting resumes from the last
# saved block of each chain.
# Plotting is run separately: pchi_plots.sh

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

SCRIPT_DIR=/home/johngallagher/data/HMC-Research/CHMC/scripts/NCC_SIAM_26
P=6
D=1200
TAU=0.072
T=4.0
OUTPUT=/scratch/johngallagher/pchi_p${P}_d${D}_tau${TAU}_blocks

echo "Running p-chi timed blocks, p=${P}, d=${D}, tau=${TAU}, T=${T}"
python -u ${SCRIPT_DIR}/timed_blocks_pchi.py --p ${P} --d ${D} --taus ${TAU} --Ts ${T} --output ${OUTPUT}
