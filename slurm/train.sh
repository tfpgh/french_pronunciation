#!/usr/bin/env bash

#SBATCH --job-name=train
#SBATCH --output=logs/%j.out
#SBATCH --partition=gpu-standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=200gb
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=2-00:00:00

export TQDM_MININTERVAL=60

uv run python -u train.py
