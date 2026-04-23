#!/usr/bin/env ipython

from pathlib import Path
from string import ascii_letters
from typing import Callable
from uuid import uuid4

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import pytest
import torch
from amr_predict.utils import (
    deduplicate,
    sample_pairs,
    smoothen_log2,
    with_external_amr_predictions,
)
from datasets import Dataset
from loguru import logger
from pyhere import here
from torch.utils.data import DataLoader

logger.enable("amr_predict")


@pytest.fixture
def rng():
    return np.random.default_rng()


list_of_arr = [np.random.rand(np.random.randint(3, 9), 10) for _ in range(10)]
fixed_arr = [np.random.rand(10) for _ in range(10)]

df = pl.DataFrame(
    {"e": fixed_arr, "t": list_of_arr, "n": None},
    schema={
        "e": pl.Array(pl.Float64, 10),
        "t": pl.List(pl.Array(pl.Float64, 10)),
        "n": pl.Null,
    },
)


def test_with_external_amr(env):
    env["seq_metadata"]["hamronization"] = here(
        "data",
        "remote",
        "output/ast_browser/funcscan/batch1/reports/hamronization_summarize/hamronization_combined_report.tsv",
    )
    env["ast_metadata"]["binarize"] = False
    df = pl.read_csv(here("data", "meta", "ast_subsampled.tsv"), separator="\t").rename(
        {"BioSample": "sample"}
    )
    result = with_external_amr_predictions(df, env)
    for col in result.columns:
        if col.endswith("_cm"):
            print(result[col].value_counts())


def dummy_df() -> pl.DataFrame:
    dummy_df = pl.DataFrame(
        {"a": list(ascii_letters), "b": torch.rand(len(ascii_letters))}
    )

    pl.LazyFrame(
        {"a": "a", "b": [torch.rand(9)]},
        schema={"a": pl.String, "b": pl.Array(pl.Float64, 9)},
    ).collect()

    dummy_df.with_columns(
        pl.when(
            pl.Series(rng().choice([True, False], p=[0.3, 0.7], size=dummy_df.height))
        )
        .then(pl.lit(None))
        .otherwise(pl.col("a"))
        .alias("a")
    )["a"].value_counts()

    dummy_df.filter(rng().choice([True, False], p=[0.3, 0.7], size=dummy_df.height))
    return dummy_df


def test_deduplicate():
    df = pl.DataFrame({"key": list("aaabbccddfe"), "x": np.random.random(11)})
    dset = Dataset.from_polars(df)
    deduplicated = deduplicate(dset, "key")
    assert len(deduplicated["key"]) == 6


def test_smoothen():
    vals = np.array([3.82e-03, 7.8125e-03, 3.225e-02, 5.12e2, 2.1])
    expected = np.array([3.90625e-03, 7.8125e-03, 3.125e-2, 5.12e2, 2])
    smoothened = smoothen_log2(vals)
    assert (smoothened == expected).all()


def test_sample_pairs():
    tmp = {"var": list("A" * 50 + "B" * 50 + "C" * 50 + "D" * 50 + "E" * 10)}
    df = pd.DataFrame(tmp)
    n_checks = 10
    for _ in range(n_checks):
        related, unrelated = sample_pairs(
            df, var="var", n_pairs_per=20, id_col=None, replace=False
        )
        assert (
            df.iloc[related[:, 0]]["var"].values == df.iloc[related[:, 1]]["var"].values
        ).all()
        assert (
            df.iloc[unrelated[:, 0]]["var"].values
            != df.iloc[unrelated[:, 1]]["var"].values
        ).all()


# cache = EmbeddingCache(
#     Path(
#         "/home/shannc/Bio_SDD/amr_predict/results/tests/seqlens_test/datasets/embedded/orf_only_seqLens_cache"
#     )
# )
# dset = load_as(
#     "/home/shannc/Bio_SDD/amr_predict/results/tests/seqlens_test/datasets/processed_sequences/orf_only",
# )
# ld = LinkedDataset(dset.select_columns(["sample", "seqid", "sequence"]), cache)


# loader = td.DataLoader(ld, batch_size=6, collate_fn=None)
# l2 = td.DataLoader(
#     dset.select_columns(["sample", "seqid", "sequence"]), batch_size=6, collate_fn=None
# )
# batch = next(iter(loader))
