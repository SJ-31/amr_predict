#!/usr/bin/env ipython

from itertools import batched
from pathlib import Path

import gget
import polars as pl
from amr_predict.utils import read_tabular
from loguru import logger
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

seen_file = workdir / "seen_already.txt"
failed_file = workdir / "failed.txt"

SEEN = set()
if seen_file.exists():
    SEEN = set(seen_file.read_text().strip().split("\n"))
FAILED = set()
if failed_file.exists():
    FAILED = set(failed_file.read_text().strip().split("\n"))

ids = set(has_ensembl["Ensembl"]) - SEEN - FAILED
batches = list(workdir.glob(pattern=".fasta$"))
start_from = len(batches)


def get_individually(batch) -> list[str]:
    result = []
    for acc in batch:
        try:
            lookup = gget.seq(acc)
            result.extend(lookup)
        except RuntimeError:
            logger.warning("Failed to get sequence for {}", acc)
            FAILED.add(acc)
    return result


if __name__ == "__main__":
    for i, batch in enumerate(batched(ids, 500)):
        batch_number = start_from + i
        outfile = workdir / f"batch_{batch_number}.fasta"
        logger.info("Retrieving batch {}", batch_number)
        logger.info("Accessions: {}", batch)
        SEEN |= set(batch)
        try:
            seqs = gget.seq(batch)
        except RuntimeError:
            seqs = get_individually(batch)
        logger.success("Finished batch {}", batch_number)
        for file, to_write in zip(
            (outfile, seen_file, failed_file), (seqs, SEEN, FAILED)
        ):
            with open(file, "w") as f:
                f.write("\n".join(to_write))
