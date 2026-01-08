#!/usr/bin/env bash

# NOTE: batch 4 on [2026-01-08 Thu]
# Change input and outdir for each batch
remote=".."
nextflow run nf-core/funcscan \
    --input ~/amr_predict/config/nf_runs/funcscan/ast_browser_d-4_2025-12-04.csv \
    --outdir "${remote}/output/ast_browser/funcscan/batch4" \
    -profile singularity \
    --run_arg_screening \
    --run_amp_screening \
    --run_bgc_screening \
    --annotation_tool bakta \
    --annotation_bakta_db "${remote}/datasets/bakta_db_v5.1_full" \
    --save_annotations true
