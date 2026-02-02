#!/usr/bin/env bash

#SBATCH --job-name=generate_embeddings
#SBATCH --output=logs/%j.out
#SBATCH --partition=gpu-standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=200gb
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --time=2-00:00:00

export TQDM_MININTERVAL=15

uv run generate_embeddings.py
