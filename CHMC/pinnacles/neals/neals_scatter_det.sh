#!/bin/bash
#SBATCH --job-name=neals_scatter_det
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=0-00:10:00
#SBATCH --output=neals_scatter_det_%j.out

# Scatter of (x_1, y) samples vs the 1 and 2 sd target level sets, HMC-LF | CHMC-AA | CHMC-Newton,
# for the log|det J| corrected run (/scratch/johngallagher/neals_funnel_det_samekeys_tau0.125_T3, same keys) and the
# uncorrected run (/scratch/johngallagher/neals_funnel_blocks), tau = 0.125, T = 3 (script defaults;
# other settings: add --tau 0.0625 to the python line). Sized for slides.
# Plots go to /home/johngallagher/data/HMC-Research/plots/neals_funnel_det/scatter/.

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

for RUN in 0 1 2; do
    echo "run $RUN"
    python -u /home/johngallagher/data/HMC-Research/CHMC/scripts/NCC_SIAM_26/funnel_scatter_det.py --run "$RUN"
done
