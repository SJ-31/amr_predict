#!/usr/bin/env bash

snakemake --use-singularity \
    --singularity-args "--nv --env pythonpath=$HOME/amr_predict/src --bind /data/project/stemcell/shannc/repos:$PWD/../../data/remote"
