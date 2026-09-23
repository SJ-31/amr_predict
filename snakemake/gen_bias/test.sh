#!/usr/bin/env bash

export HF_HOME="/data/project/stemcell/shannc/repos/amr_predict/cache/huggingface"
snakemake --use-singularity \
	--singularity-args "--nv --bind ../../src:/py_lib --bind /data/project:/data/project" \
	--configfile test_env.yaml --profile profiles/test/config.yaml \
	"$@"
