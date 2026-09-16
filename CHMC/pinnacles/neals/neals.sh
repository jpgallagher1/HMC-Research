#!/bin/bash
#SBATCH --job-name=neals
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-6:00:00
#SBATCH --array=0-5
#SBATCH --output=neals_%A_%a.out
#SBATCH --error=neals_%A_%a.err

set -e

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

SCRIPT_DIR=/home/johngallagher/data/HMC-Research/CHMC/scripts/neals

# One method per task; each task runs T=1..5 in sequence.
METHOD=${SLURM_ARRAY_TASK_ID}

for T in 1 2 3 4 5; do
case ${METHOD} in
    0)
        echo "Running AA_m=2, T=${T}"
        python -u "${SCRIPT_DIR}/chmc_neals_AA.py" 2 "${T}"
        ;;
    1)
        echo "Running AA_m=3, T=${T}"
        python -u "${SCRIPT_DIR}/chmc_neals_AA.py" 3 "${T}"
        ;;
    2)
        echo "Running AA_m=4, T=${T}"
        python -u "${SCRIPT_DIR}/chmc_neals_AA.py" 4 "${T}"
        ;;
    3)
        echo "Running FPI, T=${T}"
        python -u "${SCRIPT_DIR}/chmc_neals_FPI.py" "${T}"
        ;;
    4)
        echo "Running LF, T=${T}"
        python -u "${SCRIPT_DIR}/chmc_neals_LF.py" "${T}"
        ;;
    5)
        echo "Running Newton, T=${T}"
        python -u "${SCRIPT_DIR}/chmc_neals_Newton.py" "${T}"
        ;;
esac
done