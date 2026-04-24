#!/usr/bin/env bash

# WARNING: don't use this script if you're using the embedding models

srun --job-name=prepare_data \
    --qos=cpu24h \
    --mem=60G \
    snakemake -s prepare_data.smk \
    --profile profiles/default \
    "$@"
