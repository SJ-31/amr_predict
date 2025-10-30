#!/usr/bin/env bash

paste <(ls *_1.fastq.gz | sed 's/_1.fastq.gz//' | sort) <(find "$(
    cd ..
    pwd
)" -name "*_1.fastq.gz" | sort) <(find "$(
    cd ..
    pwd
)" -name "*_2.fastq.gz" | sort) >"$1"
