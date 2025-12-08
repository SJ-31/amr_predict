#!/usr/bin/env ipython

from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest
import tomllib
from amr_predict.preprocessing import SeqDataset, SeqEmbedder, SeqPreprocessor
from amr_predict.utils import split_features
from Bio import SeqIO
from datasets import Dataset, disable_caching, load_from_disk
from loguru import logger
from pyhere import here

disable_caching()
# %%

with open(here("tests", "env.toml"), "rb") as f:
    ENV: dict = tomllib.load(f)

logger.enable("amr_predict")

# * Utilities


def read_fasta(file):
    return {record.name: record for record in SeqIO.parse(file, "fasta")}


def get_fastas():
    tmp = Path(ENV["fastas"])
    return {f.stem: read_fasta(f) for f in tmp.iterdir() if f.suffix == ".fasta"}


def get_diffs(df, feature_col: str = "locus_tag"):
    grouped = (
        df.group_by(feature_col)
        .agg(
            pl.col("old_start").first(),
            pl.col("old_stop").first(),
            pl.col("start").min(),
            pl.col("stop").max(),
        )
        .with_columns(
            start_diff=pl.col("old_start") - pl.col("start"),
            stop_diff=pl.col("stop") - pl.col("old_stop"),
        )
    )
    return grouped


@pytest.fixture
def bin_dataset() -> Dataset:
    ds: Dataset = load_from_disk(ENV["bindata"])
    return ds


FASTAS = get_fastas()


# * Tests


def test_add_utrs():
    # For bakta "Locus Tag" uniquely identifies observations
    # Expect that both starts and stops have been expanded, and that the difference between
    # old and new is no greater than `utr_amount`
    sep = SeqPreprocessor(
        seq_path=Path(ENV["fastas"]),
        anno_path=Path(ENV["bakta"]),
        split_method="bakta",
        max_length=300,
        utr_amount=(100, 100),
        upstream_context=0,
        downstream_context=0,
    )
    with TemporaryDirectory() as tmp:
        dataset = Dataset.from_generator(sep.gen, cache_dir=tmp).to_polars()
    diffs = get_diffs(dataset, "locus_tag")
    assert all(diffs["start_diff"] > 0)
    assert all(diffs["start_diff"] <= 100)
    assert all(diffs["stop_diff"] >= 0)
    assert all(diffs["stop_diff"] <= 100)
    return dataset


def test_split_features():
    sample_bakta = pl.read_csv(
        next(Path(ENV["bakta"]).iterdir()),
        separator="\t",
        skip_rows=5,
        infer_schema_length=None,
    )
    split = split_features(sample_bakta, 120, "Start", "Stop")
    assert all((split["chunk_Stop"] - split["chunk_Start"]) <= 120)


def test_kmer_embed(tmp_path):
    feature_file: Path = tmp_path / "features.txt"
    emb = SeqEmbedder(
        method="kmer",
        fastas=Path(ENV["fastas"]),
        id_col="sample",
        k=3,
        key="x",
        save_features_to=feature_file,
        var_quantile_threshold=0.5,
    )
    embedded = emb(None)
    features = feature_file.read_text().splitlines()
    assert len(features) == 32 and "AAA" in features
    emb2 = SeqEmbedder(
        method="kmer",
        fastas=Path(ENV["fastas"]),
        id_col="sample",
        k=3,
        key="x",
        features=features,
        var_quantile_threshold=0.5,
    )
    e2 = emb2(None)


def test_embed_hamr(tmp_path):
    feature_file: Path = tmp_path / "features.txt"
    kws = dict(
        method="feature_presence",
        fasta_annotations=here("results", "tests", "no_date", "seq_metadata.csv"),
        feature_cols=["hamr_gene_symbol", "hamr_resistance_mechanism"],
        id_col="sample",
        key="x",
        save_features_to=feature_file,
        var_quantile_threshold=0.5,
    )
    emb = SeqEmbedder(**kws)
    embedded = emb(None)
    features = feature_file.read_text().splitlines()
    print(features[1:10])
    del kws["save_features_to"]
    kws["features"] = features
    emb2 = SeqEmbedder(**kws)
    e2 = emb2(None)


def test_embed_bakta(tmp_path):
    feature_file: Path = tmp_path / "features.txt"
    kws = dict(
        method="feature_presence",
        fasta_annotations=Path(ENV["bakta"]),
        feature_cols=["Gene", "Product"],
        id_col="sample",
        key="x",
        save_features_to=feature_file,
        var_quantile_threshold=0.5,
        read_kws={"comment_prefix": "# "},
        metadata_pattern="*_bakta.tsv",
    )
    emb = SeqEmbedder(**kws)
    embedded = emb(None)
    features = feature_file.read_text().splitlines()
    print(features[1:10])
    del kws["save_features_to"]
    kws["features"] = features
    emb2 = SeqEmbedder(**kws)
    e2 = emb2(None)


def test_utrs2():
    sep = SeqPreprocessor(
        seq_path=Path(ENV["fastas"]),
        anno_path=Path(ENV["bakta"]),
        split_method="bakta",
        max_length=125,
        utr_amount=(200, 0),
    )
    # Max length for any sequence should be 225 (125 + 50 + 50)
    with TemporaryDirectory() as tmp:
        df = Dataset.from_generator(sep.gen, cache_dir=tmp).to_polars()
    df = df.with_columns(seqlen=pl.col("sequence").str.len_chars()).with_columns(
        diff=pl.col("stop") - pl.col("start")
    )
    others = df.filter(~pl.col("is_5prime"))
    fivep = df.filter(pl.col("is_5prime"))
    assert all(fivep["seqlen"] >= 125)
    assert all(others["seqlen"] <= 125)
    return df


def test_add_context():
    sep = SeqPreprocessor(
        seq_path=Path(ENV["fastas"]),
        anno_path=Path(ENV["bakta"]),
        split_method="bakta",
        max_length=125,
        utr_amount=None,
        upstream_context=50,
        downstream_context=50,
    )
    # Max length for any sequence should be 225 (125 + 50 + 50)
    with TemporaryDirectory() as tmp:
        df = Dataset.from_generator(sep.gen, cache_dir=tmp).to_polars()
    df = df.with_columns(seqlen=pl.col("sequence").str.len_chars()).with_columns(
        diff=pl.col("stop") - pl.col("start")
    )
    assert all(df["seqlen"] <= 225)
    return df
