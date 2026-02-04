#!/usr/bin/env bash

#SBATCH --job-name=train
#SBATCH --output=logs/%j.out
#SBATCH --partition=gpu-standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=300gb
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --time=2-00:00:00

export TQDM_MININTERVAL=60
export PYTHONUNBUFFERED=1

uv run torchrun --nproc_per_node=4 train.py
