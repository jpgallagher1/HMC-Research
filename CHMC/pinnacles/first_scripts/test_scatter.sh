#!/bin/bash
#SBATCH --nodes=1    # request only 1 node
#SBATCH --partition test      # this job will be submitted to test queue
#SBATCH --mem=32G #this job is asked for 96G of total memory, use 0 if you want to use entire node memory
#SBATCH --time=0-00:15:00 # 15 minute
#SBATCH --ntasks-per-node=56 # this job requests for 56 cores on a node
#SBATCH --output=my_%j.stdout    # standard output will be redirected to this file
# #SBATCH --constraint=bigmem   #uncomment this line if you need the access to the bigmem node for Pinnacles
# #SBATCH --constraint=gpu #uncomment this line if you need the access to GPU
# #SBATCH --gres=gpu:2   #uncomment this line if you need GPU access (2 GPUs)
#SBATCH --job-name=my_other_job    # this is your job’s name
# #SBATCH --mail-user=johngallagher@ucmerced.edu  
# #SBATCH --mail-type=ALL  #uncomment the first two lines if you want to receive     the email notifications
#SBATCH --export=ALL


python CHMC/scripts/test_scatter.py