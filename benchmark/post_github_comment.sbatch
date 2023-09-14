#!/bin/bash
#SBATCH --job-name=trl
#SBATCH --partition=production-cluster
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out

sleep 2m
bash benchmark/plot_core.sh
srun python benchmark/post_github_comment.py
