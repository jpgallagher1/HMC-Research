#!/bin/bash
#SBATCH --job-name=mcmc_avg   # this is your job’s name
# #SBATCH --mail-user=johngallagher@ucmerced.edu  
# #SBATCH --mail-type=ALL  #uncomment the first two lines if you want to receive     the email notifications

#SBATCH --nodes=1    
#SBATCH --partition=short     # partition name
#SBATCH --mem=8G #this job is asked for 96G of total memory, use 0 if you want to use entire node memory
#SBATCH --time=0-00:15:00 # 15 minute
#SBATCH --ntasks-per-node=3 # this job requests for 3 cores on a node
# #SBATCH --output=mcmc%j.stdout    # standard output will be redirected to this file

#SBATCH --export=ALL


OUTPUT_DIR="mcmc_results/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

for i in 1 2 3; do
    python CHMC/scripts/indep_runs.py $i --output "$OUTPUT_DIR/chainavg_$i.npy" &
done
wait