#!/bin/bash
#SBATCH --job-name=trl
#SBATCH --partition=production-cluster
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out

cleanup () {
    rm -rf "$WORK_DIR"
    echo "Deleted temp working directory $WORK_DIR"
}
trap cleanup EXIT

sleep 2m
bash $BENCHMARK_PLOT_SCRIPT
srun python benchmark/post_github_comment.py
