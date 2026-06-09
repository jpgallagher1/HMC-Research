#!/bin/bash
#SBATCH --job-name=mcmc_scaling
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20        # max needed across loop
#SBATCH --partition=short
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --output=mcmc_scaling%j.stdout

OUTPUT_DIR="mcmc_results/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

RESULTS_FILE="${OUTPUT_DIR}/scaling_results.txt"
echo "Cores | Time (s) | Speedup | Efficiency (%)" > $RESULTS_FILE
echo "------|----------|---------|----------------" >> $RESULTS_FILE

T_SERIAL=""

for CORES in 1 2 4 10 20
do
    echo "Running with $CORES cores..."
    LOG_FILE="${OUTPUT_DIR}/run_${CORES}.log"
    
    # srun dynamically allocates cores per iteration
    # XLA_FLAGS tells JAX to use them as devices
    srun --ntasks=1 --nodes=1 --cpus-per-task=$CORES \
        --export=ALL,XLA_FLAGS="--xla_force_host_platform_device_count=$CORES" \
        python CHMC/scripts/scaling_study_hi_dim.py 2>&1 | tee "$LOG_FILE"
    
    # Use sampling time from Python output
    T_N=$(grep "Sampling time:" "$LOG_FILE" | awk '{print $3}' | sed 's/s//')
    
    if [ -z "$T_SERIAL" ]; then
        T_SERIAL=$T_N
    fi
    
    SPEEDUP=$(echo "scale=2; $T_SERIAL / $T_N" | bc)
    EFFICIENCY=$(echo "scale=1; ($SPEEDUP / $CORES) * 100" | bc)
    
    echo "$CORES     | $T_N    | $SPEEDUP    | $EFFICIENCY%" | tee -a $RESULTS_FILE
done

echo ""
echo "=== Scaling Study Complete ==="
cat $RESULTS_FILE