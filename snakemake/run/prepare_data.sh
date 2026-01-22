#!/usr/bin/env bash

srun --job-name=prepare_data \
    --nodes=1 \
    --cpus-per-task=1 \
    --mem-per-cpu=30GB \
    --qos=gpu20gh \
    --partition=gpu \
    --gres=gpu:3g.20gb:1 \
    snakemake -s prepare_data.smk \
    --profile profiles/default \
    "$@"
