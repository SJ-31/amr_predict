#!/usr/bin/env bash

workdir="../../data/remote/bakta_mmseqs_2026-03-25/"
cd "${workdir}"

# Activate mmseqs environment

srun --qos=cpu24h --mem=30G mmseqs createdb seqs.fasta mmdb/

# Full-length coverage clustering at 0.95 sequence overlap
srun --qos=cpu24h --mem=80G mmseqs cluster \
    -c 0.95 \
    --cov-mode 0 \
    mmdb/db clustdb/db tmp

mmseqs createtsv mmdb/db mmdb/db clustdb/db clusters.tsv
