#!/bin/bash
#SBATCH --job-name=neals_Newton
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-6:00:00
#SBATCH --array=0-11
#SBATCH --output=neals_Newton_%A_%a.out

# goal is to run 12 total workers on a single core in parallel to get better 
# numbers.  The shared: 1work: 1 node, 4 core split tasks, affects the run time 
# dramatically. I dont yet know the best way to run wallclock tests. 

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

export TASKFILE=$SLURM_SUBMIT_DIR/tasks.txt
export TOTAL_WORKERS=12

# Worker number across all array jobs: 0-11 in job 0, 12-123 in job 1, ...
# Each worker runs every 12th line, starting at its own offset
srun bash -c '
WORKER=$((SLURM_ARRAY_TASK_ID * SLURM_NTASKS + SLURM_PROCID))
sed --quiet "$((WORKER+1))~$TOTAL_WORKERS p" "$TASKFILE" | while read -r cmd; do $cmd; done
'