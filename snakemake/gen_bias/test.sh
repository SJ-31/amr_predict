#!/usr/bin/env bash

snakemake --use-singularity \
	--singularity-args "--nv --bind ../../src:/py_lib --bind /data/project:/data/project --cleanenv" \
	--configfile test_env.yaml --profile profiles/test/config.yaml \
	"$@"
