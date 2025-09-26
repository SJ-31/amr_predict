#!/usr/bin/env bash

# [2025-09-22 Mon] Keep it using the jia samples for now
# Run from repos/amr_predict/pipelines
# [2025-09-24 Wed] Disabled protein annotation due to errors
remote=".."
nextflow run nf-core/funcscan \
    --input "./jia_samplesheet.csv" \
    --outdir "${remote}/output/jia_funcscan_2025-09-22" \
    -profile singularity \
    --run_arg_screening \
    --run_amp_screening \
    --run_bgc_screening \
    --annotation_tool bakta \
    --annotation_bakta_db "${remote}/datasets/bakta_db_v5.1_full" \
    --save_annotations true
