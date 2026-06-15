#!/bin/bash
#SBATCH --job-name=mcmc_scaling   
# #SBATCH --mail-user=johngallagher@ucmerced.edu  
# #SBATCH --mail-type=ALL  

#SBATCH --nodes=1    
#SBATCH --partition=short     # partition name
#SBATCH --mem=16G #this job is asked for 16G of total memory, use 0 if you want to use entire node memory
#SBATCH --time=0-00:30:00 # 30 minute
#SBATCH --output=mcmc_scaling%j.stdout    # standard output will be redirected to this file
#SBATCH --cpus-per-task=10

OUTPUT_DIR="mcmc_results/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

RESULTS_FILE="${OUTPUT_DIR}/scaling_results.txt"
echo "Cores | Time (s) | Efficiency (%)" > $RESULTS_FILE
echo "------|----------|----------------" >> $RESULTS_FILE

T_SERIAL=""

for CORES in 1 2 4 10
do
    echo "Running MCMC with $CORES cores..."

    START=$(date +%s%N)
    
    python CHMC/scripts/scaling_study3.py 

    
    END=$(date +%s%N)

    # Elapsed time in seconds (floating point)
    T_N=$(echo "scale=3; ($END - $START) / 1000000000" | bc)

    # Store serial time on first iteration
    if [ -z "$T_SERIAL" ]; then
        T_SERIAL=$T_N
    fi

    # Efficiency = (T_serial / (N * T_N)) * 100
    EFFICIENCY=$(echo "scale=1; ($T_SERIAL / ($CORES * $T_N)) * 100" | bc)

    echo "$CORES     | $T_N       | $EFFICIENCY%" | tee -a $RESULTS_FILE
done

echo ""
echo "=== Scaling Study Complete ==="
cat $RESULTS_FILE