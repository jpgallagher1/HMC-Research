#!/bin/bash
#SBATCH --job-name=neals
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=0-4:00:00
#SBATCH --array=0-5
#SBATCH --output=neals_%A_%a.out

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

SCRIPT_DIR=/home/johngallagher/data/HMC-Research/CHMC/scripts/neals

case ${SLURM_ARRAY_TASK_ID} in
    0)
        echo "Running AA_m=2"
        python -u ${SCRIPT_DIR}/chmc_neals_AA.py 2
        ;;
    1)
        echo "Running AA_m=3"
        python -u ${SCRIPT_DIR}/chmc_neals_AA.py 3
        ;;
    2)
        echo "Running AA_m=4"
        python -u ${SCRIPT_DIR}/chmc_neals_AA.py 4
        ;;
    3)
        echo "Running FPI"
        python -u ${SCRIPT_DIR}/chmc_neals_FPI.py
        ;;
    4)
        echo "Running LF"
        python -u ${SCRIPT_DIR}/chmc_neals_LF.py
        ;;
    5)
        echo "Running Newton"
        python -u ${SCRIPT_DIR}/chmc_neals_Newton.py
        ;;
esac