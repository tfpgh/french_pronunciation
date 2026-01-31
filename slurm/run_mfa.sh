#!/usr/bin/env bash

#SBATCH --job-name=run_mfa
#SBATCH --output=logs/%j.out
#SBATCH --partition=standard
#SBATCH --cpus-per-task=48
#SBATCH --mem=200gb
#SBATCH --nodelist=node021
#SBATCH --time=2-00:00:00

podman pull mmcauliffe/montreal-forced-aligner:latest

mkdir -p /storage/tpenner/mfa_data

podman run --rm -it \
    --userns=keep-id \
    -v /storage/penner:/storage/tpenner \
    -e MFA_ROOT_DIR=/storage/tpenner/mfa_data \
    docker.io/mmcauliffe/montreal-forced-aligner:latest \
    mfa model download dictionary french_mfa

podman run --rm -it \
    --userns=keep-id \
    -v /storage/penner:/storage/tpenner \
    -e MFA_ROOT_DIR=/storage/tpenner/mfa_data \
    docker.io/mmcauliffe/montreal-forced-aligner:latest \
    mfa model download acoustic french_mfa
