#!/usr/bin/env bash
# Change input and outdir for each batch

# NOTE: batch 6 on [2026-01-26 Mon]
remote=".."
nextflow run nf-core/funcscan \
    --input ~/amr_predict/config/nf_runs/funcscan/ast_browser_d-6_bacass_2026-01-26.csv \
    --outdir "${remote}/output/ast_browser/funcscan/batch6" \
    -profile singularity \
    --run_arg_screening \
    --run_amp_screening \
    --run_bgc_screening \
    --annotation_tool bakta \
    --annotation_bakta_db "${remote}/datasets/bakta_db_v5.1_full" \
    --save_annotations true
