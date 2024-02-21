#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=../outputs/eval-debug.out
#SBATCH --error=../outputs/eval-debug.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --export=ALL
#SBATCH --exclude=g0001,g0028,g0018,g0022
srun -l example/evaluate.sh
echo "Done with job $SLURM_JOB_ID"