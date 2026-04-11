#!/usr/bin/env python3

from amr_predict.preprocessing import SeqEmbedder
from datasets.arrow_dataset import Dataset

try:
    from snakemake.script import snakemake as smk
except ImportError:
    smk = type("snakemake", (), {"rule": None, "config": {}, "log": [0]})

RCONFIG = smk.config.get(smk.rule)
RNG: int = smk.config.get("rng", 20021031)

logger.enable("")
if len(smk.log) == 1:
    logger.add(smk.log[0])

# * Utility functions

# def read_uniprot_fasta()

# * Rules

def make_seq_dataset():
    # dset: Dataset = 
