#!/usr/bin/env bash

workdir="../../data/remote/bakta_mmseqs_2026-03-25/"
cd "${workdir}"

# Activate mmseqs environment

# srun --qos=cpu24h --mem=30G mmseqs createdb seqs.fasta mmdb/
# srun --qos=cpu24h --mem=80G mmseqs cluster mmdb clustdb/ tmp -c 0.95
# mmseqs createtsv mmdb/ mmdb/ clustdb/ clusters.tsv
