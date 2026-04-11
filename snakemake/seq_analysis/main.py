#!/usr/bin/env python3

from pathlib import Path
from typing import Literal

import polars as pl
from amr_predict.preprocessing import EMBEDDING_METHODS, SeqEmbedder
from beartype import beartype
from Bio import SeqIO
from datasets import load_from_disk
from datasets.arrow_dataset import Dataset
from loguru import logger

try:
    from snakemake.script import snakemake as smk
except ImportError:
    smk = type("snakemake", (), {"rule": None, "config": {}, "log": [0]})

CONFIG = smk.config
RCONFIG = smk.config.get(smk.rule)
RNG: int = smk.config.get("rng", 20021031)

logger.enable("")
if len(smk.log) == 1:
    logger.add(smk.log[0])

# * Utility functions


@beartype
def read_fasta(file: str, header_style: Literal["uniprot"]) -> pl.DataFrame:
    tmp = {"id": [], "sequence": []}
    for record in SeqIO.parse(file, "fasta"):
        if header_style == "uniprot":
            try:
                id = record.id.split("|")[1]
            except IndexError:
                raise ValueError(
                    f"FASTA entry {record} does not have a UniProt-style header"
                )
        else:
            raise ValueError("Header style not recognized")

        tmp["id"].append(id)
        tmp["sequence"].append(record.seq)
    df: pl.DataFrame = pl.DataFrame(tmp)
    if not df.select("id").is_duplicated().any():
        logger.warning(f"Fasta file {file} has duplicate ids")
        df = df.unique("id")
    return df


# * Rules


def make_seq_dataset():
    dfs: pl.DataFrame = []
    outdir = Path(smk.config["outdir"])
    for spec in CONFIG["fastas"][smk.params["seqtype"]]:
        dfs.append(read_fasta(spec["file"], spec["header_style"]))
    combined = pl.concat(dfs)
    for col in ("id", "sequence"):
        if not combined[col].is_duplicated().any():
            logger.warning(f"Duplicate {col} present in fastas")
            dups: pl.DataFrame = combined.filter(combined[col].is_duplicated())
            dups.write_csv(outdir / f"duplicated_{col}_{smk.params["seqtype"]}.csv")
            combined = combined.unique(col)
    dset: Dataset = Dataset.from_polars(combined)
    dset.save_to_disk(dataset_path=smk.output[0])


def get_embeddings():
    dset: Dataset = load_from_disk(smk.input[0])
    seqtype = smk.params["seqtype"]
    spec: dict = CONFIG[seqtype][smk.params["embedding_method"]]
    method: EMBEDDING_METHODS = spec.get("method", smk.params["embedding_method"])
    kws: dict = spec.get("kws")
    out = Path(smk.output)
    cache_path = out.with_suffix("")
    if not cache_path.exists():
        cache_path.mkdir()
    if method in ("seqLens", "esm"):
        kws["huggingface"] = CONFIG["huggingface"]
    if method == "esm":
        kws["from_nucleotide"] = False
    embedder = SeqEmbedder(
        workdir=cache_path, only_cache=True, with_tokens=False, **kws
    )
    embedder(dset)
    out.write_text("completed")


# * Entry
if rule_fn := globals().get(smk.rule):
    rule_fn()
