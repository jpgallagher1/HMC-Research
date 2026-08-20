#!/bin/bash
#SBATCH --job-name=pgauss_d10
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=0-6:00:00

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research
python -u CHMC/scripts/gen_gauss/chmc_hi_dim_gen_gauss_AA_dim.py 4 100