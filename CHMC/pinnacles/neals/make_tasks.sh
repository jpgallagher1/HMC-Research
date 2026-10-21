#!/bin/bash
SCRIPT=/home/johngallagher/data/HMC-Research/CHMC/scripts/neals/chmc_neals_Newton.py
BASE=/scratch/johngallagher/neals_results/Newton
TAUS=(0.5 0.25 0.125 0.0625 0.03125)
LENS=($(ls "$BASE/tau_0.5/T_1.0" | sed 's/len_//' | sort -n))

# One line per missing run
for T in 1 2 3 4 5; do
  for TAU_IDX in 0 1 2 3 4; do
    for LEN_IDX in 0 1 2 3 4 5 6 7 8; do
      for RUN in 0 1 2 3 4 5 6 7 8 9; do
        FILE="$BASE/tau_${TAUS[$TAU_IDX]}/T_${T}.0/len_${LENS[$LEN_IDX]}/run_${RUN}.npz"
        [ -f "$FILE" ] || echo "python -u $SCRIPT $T $TAU_IDX $LEN_IDX $RUN"
      done
    done
  done
done > tasks.txt