#!/usr/bin/env ipython

from pathlib import Path

import gget
from amr_predict.utils import read_tabular
from pyhere import here

remote = here("data", "remote")

dset = read_tabular(remote / "datasets" / "2026-04-21_uniprot_swissprot_goa.tsv")

workdir: Path = remote / "datasets" / "2026-04-21_uniprot_swissprot_goa_nucleotides"
workdir.mkdir(exist_ok=True)

has_ensembl = (
    dset.filter(pl.col("Ensembl").is_not_null())
    .with_columns(pl.col("Ensembl").str.split(";"))
    .explode("Ensembl")
    .filter(pl.col("Ensembl").str.len_chars() > 0)
    .unique("Ensembl")
)

has_ensembl.select()

output_file = workdir / "nucleotides.parquet"

# TODO: [2026-04-21 Tue] finish this up
if __name__ == "__main__":
    if output_file.exists():
        df = 
    leftover = 
    
