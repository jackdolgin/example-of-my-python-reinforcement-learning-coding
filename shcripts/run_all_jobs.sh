#!/bin/bash

#SBATCH --cpus-per-task 1

ITERATIONS=10
EXPERIMENT="second_go"
python ../py_scripts/prep_params_for_cluster.py "$ITERATIONS" "$EXPERIMENT"

NUM_SUBS=$(ls ../../Data/second_go/Spliced | wc -l)
NUM_RUNS=$(($ITERATIONS * $NUM_SUBS))
sbatch --array=1-"$NUM_RUNS" run_one_job.sh