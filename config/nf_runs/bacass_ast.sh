#!/usr/bin/env bash

# [2025-12-03 Wed] TODO: run this thing

remote=~/amr_predict/data/remote
nextflow run nf-core/bacass \
    -resume \
    --input ~/amr_predict/config/nf_runs/ast_browser_samplesheet_2025-12-03.csv \
    --kraken2db "/data/project/stemcell/shannc/repos/amr_predict/cache/k2_standard_16_GB_20250714.tar.gz" \
    --ncbi_assembly_metadata "/data/project/stemcell/shannc/repos/amr_predict/cache/assembly_summary_refseq_2025-10-06.txt" \
    --kmerfinderdb "/data/project/stemcell/shannc/repos/amr_predict/cache/20190108_kmerfinder_stable_dirs.tar.gz" \
    --outdir "${remote}/output/ast_browser_bacass_2025-12-03" \
    -profile singularity \
    --skip_annotation \
    --skip_kmerfinder \
    -c "./nf_config_fix.config"
