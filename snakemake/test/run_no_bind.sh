#!/usr/bin/env bash

snakemake --use-singularity \
    --singularity-args "--nv --bind /data/project/stemcell/shannc/repos:$PWD/../../data/remote"
