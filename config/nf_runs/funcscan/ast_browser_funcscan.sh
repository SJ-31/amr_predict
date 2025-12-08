#!/usr/bin/env bash

# NOTE: starting with batch 1 [2025-12-04 Thu]
# Change input and outdir for each batch
remote=".."
nextflow run nf-core/funcscan \
    --input ~/amr_predict/config/nf_runs/funcscan/ast_browser_d-1_2025-12-04.csv \
    --outdir "${remote}/output/ast_browser/funcscan/batch1" \
    -profile singularity \
    --run_arg_screening \
    --run_amp_screening \
    --run_bgc_screening \
    --annotation_tool bakta \
    --annotation_bakta_db "${remote}/datasets/bakta_db_v5.1_full" \
    --save_annotations true
