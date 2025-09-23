#!/usr/bin/env bash

# Make genome samplesheet from the genome directory, for use with nf-core/funscan
output="$1"
if [[ -z "$2" ]]; then
    dir="genomes"
else
    dir="$1"
fi

echo "sample,fasta" >"${output}"
find "$PWD/${dir}" -name "*fasta" | sed -r 's/.*\/(.*).fasta/\1,\0/' | tail -n +2 >>"${output}"
