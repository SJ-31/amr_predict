#!/usr/bin/env bash

snakemake --use-singularity \
    --singularity-args "--nv --bind $HOME/amr_predict/src:/py_lib --bind /data/project/stemcell/shannc/repos:$PWD/../../data/remote"
