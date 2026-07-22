#!/bin/bash
for mode in "standard" "extra_npoints" "extra_npoints_extra_sigma" "known_u" "known_u_known_sigma"; do
for n in 3 5 10; do
  for s in 0.001 0.01 0.1 0.2; do
    JOB_NAME="Ex2_n${n}_sigma${s}_${r}"
    sbatch --job-name=${JOB_NAME} run_GT.slurm $n $s $mode 
    sbatch --job-name=${JOB_NAME} run_PIGP.slurm $n $s $mode
  done  
done
done
