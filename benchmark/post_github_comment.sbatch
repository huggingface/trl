#!/bin/bash
#SBATCH --job-name=trl
#SBATCH --partition=hopper-cpu
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out

sleep 2m
bash $BENCHMARK_PLOT_SCRIPT
srun python benchmark/post_github_comment.py
