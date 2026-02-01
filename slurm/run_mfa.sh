#!/usr/bin/env bash

#SBATCH --job-name=run_mfa
#SBATCH --output=logs/%j.out
#SBATCH --partition=standard
#SBATCH --cpus-per-task=48
#SBATCH --mem=200gb
#SBATCH --nodelist=node021
#SBATCH --time=2-00:00:00

MAMBA_EXE="/storage/tpenner/bin/micromamba"
MAMBA_ROOT="/storage/tpenner/micromamba_root"
export MAMBA_ROOT_PREFIX=$MAMBA_ROOT

DATA_DIR="/storage/tpenner/french_pronunciation_dataset"
OUT_DIR="/storage/tpenner/french_pronunciation_mfa_output"
TEMP_DIR="/storage/tpenner/mfa_temp"

$MAMBA_EXE run -n mfa \
    mfa align \
    $DATA_DIR/test \
    french_mfa \
    french_mfa \
    $OUT_DIR/test \
    --g2p_model_path french_mfa \
    -j 46 \
    -t $TEMP_DIR \
    --clean \
    --use_mp

$MAMBA_EXE run -n mfa \
    mfa align \
    $DATA_DIR/train \
    french_mfa \
    french_mfa \
    $OUT_DIR/train \
    --g2p_model_path french_mfa \
    -j 46 \
    -t $TEMP_DIR \
    --clean \
    --use_mp
