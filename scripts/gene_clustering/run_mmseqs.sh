#!/usr/bin/env bash

workdir="../../data/remote/bakta_mmseqs_2026-03-25/"
cd "${workdir}"

# Activate mmseqs environment

srun --qos=cpu24h --mem=30G mmseqs createdb seqs.fasta mmdb/db

# Full-length coverage clustering at 0.95 sequence overlap
srun --qos=cpu24h --mem=80G mmseqs cluster \
    -c 0.95 \
    --cov-mode 0 \
    mmdb/db clustdb/db tmp

# [2026-03-27 Fri] completed
mmseqs createtsv mmdb/db mmdb/db clustdb/db clusters.tsv

# * Cluster protein sequences
# [2026-03-30 Mon] this is probably the better way of clustering by functionality

seqkit seq -m 3 seqs.fasta | seqkit translate -T 11 >seqs_aa.fasta # Using the Bacterial, Archaeal and Plast Plastid translation table https://www.ncbi.nlm.nih.gov/Taxonomy/taxonomyhome.html/index.cgi?chapter=tgencodes#SG11
srun --qos=cpu24h --mem=30G mmseqs createdb seqs_aa.fasta mmdb/db_aa

srun --qos=cpu24h --mem=80G mmseqs cluster \
    -c 0.98 \
    --cov-mode 0 \
    mmdb/db_aa clustdb/db_aa tmp

mmseqs createtsv mmdb/db_aa mmdb/db_aa clustdb/db_aa clusters_aa.tsv
