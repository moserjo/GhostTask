#!/bin/bash
#SBATCH --partition=IMLpcluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=./slurmfiles/slurm-%j.out
#SBATCH --error=./slurmfiles/slurm-%j.err
#SBATCH --job-name=sigmaGT

npoints=$1
sigma=$2
mode=$3

# Activate conda environment if needed
source /home/moser/anaconda3/etc/profile.d/conda.sh
conda activate VENTUS1

# Run the training script
python ./main_PIGP.py $npoints $sigma $mode
