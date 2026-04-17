#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Literal

import polars as pl
import torch
import yaml
from amr_predict.evaluation import pami_wrapper
from amr_predict.preprocessing import EMBEDDING_METHODS, SeqEmbedder
from amr_predict.utils import EmbeddingCache
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
SCOL = smk.config.get("metadata", {}).get("sample_col")
LABEL_SEP = smk.config.get("metadata", {}).get("label_sep")
LCOL = smk.config.get("metadata", {}).get("label_col")
logger.enable("")
if len(smk.log) == 1:
    logger.add(smk.log[0])

os.environ["HF_HOME"] = smk.config["huggingface"]

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
        tmp["id"].append(id)
        tmp["sequence"].append(str(record.seq))
    df: pl.DataFrame = pl.DataFrame(tmp)
    if not df.select("id").is_duplicated().any():
        logger.warning(f"Fasta file {file} has duplicate ids")
        df = df.unique("id")
    return df


def account_for_max_length(
    dset: pl.DataFrame,
    max_len: int,
    seq_col: str = "sequence",
    embedding_col: str = "x",
    id_col: str = "id",
    cache: EmbeddingCache | None = None,
    agg: Literal["max", "mean", "sum"] = "mean",
) -> pl.DataFrame:
    """
    Split sequences in `dset` into subsequences of length `max_len` so that
    they can be processed by the GLM

    Parameters
    ----------
    cache : EmbeddingCache | None
        When provided, aggregate split subsequences back into their original sequence by
        the specified method in `agg`
    """
    subseq_id_col = f"{id_col}_subseq"

    expanded: pl.DataFrame = (
        dset.with_columns(pl.col(seq_col).str.len_chars().alias("length"))
        .with_columns(
            pl.int_ranges(end="length", step=max_len).alias("slice_start"),
        )
        .with_columns(
            pl.int_ranges(end=pl.col("slice_start").list.len()).alias(subseq_id_col),
        )
        .explode("slice_start", subseq_id_col)
        .with_columns(
            pl.concat_str(id_col, "id_subseq", separator=".").alias(subseq_id_col),
            pl.col(seq_col).str.slice("slice_start", length=max_len).alias(seq_col),
        )
    )
    if cache is None:
        return expanded.select(subseq_id_col, seq_col)
    retrieved: pl.DataFrame = cache.retrieve(expanded[seq_col])
    expanded = (
        expanded.join(
            retrieved, left_on=seq_col, right_on="key", how="inner", validate="m:1"
        )
        .group_by(id_col)
        .agg(
            pl.when(agg == "max")
            .then(pl.col("seq").max())
            .when(agg == "mean")
            .then(pl.col("seq").mean())
            .otherwise(pl.col("seq").sum()),
            pl.len().alias("n_subseq"),
        )
        .rename({"seq": embedding_col})
        .select(id_col, embedding_col)
    )
    return expanded


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


def label_cooccurrence():
    from PAMI.frequentPattern.basic.FPGrowth import FPGrowth

    cuda_apriori_available = False
    cudaAprioriTID = None
    try:
        from PAMI.frequentPattern.cuda import cudaAprioriTID

        cuda_apriori_available: bool = True
    except (ModuleNotFoundError, ImportError):
        pass
    if torch.cuda.is_available() and cuda_apriori_available:
        alg = cudaAprioriTID
    else:
        alg = FPGrowth
    labels = pl.read_csv(smk.input[0])
    frequent_patterns, pattern_stats = pami_wrapper(
        labels,
        alg,
        label_col=LCOL,
        sep=LABEL_SEP,
        min_sup=smk.config.get("co_occurence_min_support", 0.3),
        tmp_file=smk.params["tmp_file"],
    )
    frequent_patterns.with_columns(pl.col("Patterns").str.join(";")).write_csv(
        smk.output[0]
    )
    with open(smk.output[1], "w") as f:
        yaml.safe_dump(f, pattern_stats)


def get_embeddings():
    df: pl.DataFrame = load_from_disk(smk.input[0]).to_polars()
    seqtype = smk.params["seqtype"]
    spec: dict = CONFIG["embedding_methods"][seqtype][smk.params["embedding_method"]]
    method: EMBEDDING_METHODS = spec.get("method", smk.params["embedding_method"])
    max_length = CONFIG["embedding_max_lengths"][method]
    df = account_for_max_length(df, max_len=max_length, seq_col="sequence", id_col="id")
    kws: dict = spec.get("kws", {})
    kws["method"] = method
    out = Path(smk.output[0])
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
    embedder(Dataset.from_polars(df))
    out.write_text("completed")


# * Entry
if rule_fn := globals().get(smk.rule):
    rule_fn()
