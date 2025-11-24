#!/usr/bin/env bash

# Run from repos/amr_predict/pipelines
# [2025-09-24 Wed] Disabled protein annotation due to errors
#
# [2025-11-06 Thu] BUG: funcscan can't handle samplesheets with 1500+ samples
# It works if you split it up into two
remote=".."

nextflow run nf-core/funcscan \
    --input "./moradigaravand_samplesheet_2.csv" \
    --outdir "${remote}/output/moradigaravand_funcscan_2025-10-24" \
    -profile singularity \
    --run_arg_screening \
    --run_amp_screening \
    --run_bgc_screening \
    --annotation_tool bakta \
    --annotation_bakta_db "${remote}/datasets/bakta_db_v5.1_full" \
    --save_annotations true \
    -c ./ignore_ampcombi.config
