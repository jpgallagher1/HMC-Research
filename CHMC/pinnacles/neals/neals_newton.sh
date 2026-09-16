#!/bin/bash
#SBATCH --job-name=neals_Newton
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-6:00:00
#SBATCH --array=0-3
#SBATCH --output=neals_Newton_%A_%a.out

cd /data/johngallagher/HMC-Research
source ~/.bashrc
conda activate HMC-Research

export TASKFILE=$SLURM_SUBMIT_DIR/tasks.txt
export TOTAL_WORKERS=$((SLURM_ARRAY_TASK_COUNT * SLURM_NTASKS))

# Worker number across all array jobs: 0-5 in job 0, 6-11 in job 1, ...
# Each worker runs every 24th line, starting at its own offset
srun bash -c '
WORKER=$((SLURM_ARRAY_TASK_ID * SLURM_NTASKS + SLURM_PROCID))
sed --quiet "$((WORKER+1))~$TOTAL_WORKERS p" "$TASKFILE" | while read -r cmd; do $cmd; done
'