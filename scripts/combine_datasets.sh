#!/usr/bin/env bash

root="/data/project/stemcell/shannc/repos/amr_predict"
funcscan_root="${root}/output/ast_browser/funcscan"

#  ran on [2026-02-23 Mon]
./combine_datasets.py -i "${root}/2026-01-13_ast_esm_b1" \
    "${root}/2026-01-22_ast_esm_b2" \
    "${root}/2026-01-23_ast_esm_b3" \
    "${root}/2026-01-31_ast_esm_b4" \
    "${root}/2026-02-03_ast_esm_b5" \
    "${root}/2026-02-06_ast_esm_b6" \
    -o "${root}/2026-02-23_ast_all/datasets"

# ran [2026-02-24 Tue]
# ./combine_datasets.py -i \
#     "${funcscan_root}/batch1" \
#     "${funcscan_root}/batch2" \
#     "${funcscan_root}/batch3" \
#     "${funcscan_root}/batch4" \
#     "${funcscan_root}/batch5" \
#     "${funcscan_root}/batch6" \
#     -o "${funcscan_root}/all" \
#     --funcscan
