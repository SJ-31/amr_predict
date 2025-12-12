#!/usr/bin/env bash

# [2025-12-05 Fri] Started
remote=~/amr_predict/data/remote
nextflow run -r master nf-core/bacass \
    -resume \
    --input ~/amr_predict/config/nf_runs/ast_browser_samplesheet_2025-12-05.csv \
    --kraken2db "/data/project/stemcell/shannc/repos/amr_predict/cache/k2_standard_16_GB_20250714.tar.gz" \
    --kmerfinderdb "/data/project/stemcell/shannc/repos/amr_predict/cache/20190108_kmerfinder_stable_dirs.tar.gz" \
    --outdir "${remote}/output/ast_browser/bacass_2025-12-05" \
    -profile singularity \
    --skip_annotation \
    --skip_kmerfinder \
    --assembly_type "short" \
    -c "./nf_config_fix.config"

# --ncbi_assembly_metadata "/data/project/stemcell/shannc/repos/amr_predict/cache/assembly_summary_refseq_2025-10-06.txt" \ [2025-12-11 Thu] This flag is not used in latest version
