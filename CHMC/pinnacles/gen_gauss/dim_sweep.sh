#!/bin/bash
#SBATCH --job-name=Hidimsweep
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=0-10:00:00

cd /data/johngallagher/HMC-Research
python -u CHMC/scripts/gen_gauss/chmc_hi_dim_gen_gauss1280:40960.py 7 12 6