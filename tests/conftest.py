#!/usr/bin/env ipython

from collections.abc import Sequence
from pathlib import Path
from string import ascii_uppercase
from typing import Callable, Literal, TypeAlias, get_args

import mimesis
import numpy as np
import polars as pl
import pytest
import tomllib
import torch
import yaml
from amr_predict.cache import EmbeddingCache, LinkedDataset
from amr_predict.enums import BasicPoolings
from datasets import Dataset
from pyhere import here


@pytest.fixture
def env():
    with open(here("tests", "env.toml"), "rb") as f:
        env: dict = tomllib.load(f)

    with open(here("snakemake", "amr", "env.yaml"), "rb") as f:
        env.update(yaml.safe_load(f))
    td = here("tests", "data")
    env["ast_metadata"]["file"] = here("data", "meta", "ncbi_all_samples.tsv")
    env["seq_metadata"]["hamronization"] = td / "hamronization_combined_report.tsv"
    env["seq_metadata"]["bakta"] = td / "bakta"
    env["seq_metadata"]["ampcombi"] = td / "combgc"
    env["seq_metadata"]["combgc"] = td / "Ampcombi_summary_cluster.tsv"
    return env


@pytest.fixture
def remote():
    return here("data", "remote")


@pytest.fixture
def keys(env):
    return (
        env["pool_embeddings"]["key"],
        env["pool_embeddings"]["sample_key"],
    )


@pytest.fixture
def rdset(remote):
    SET_NAMES: TypeAlias = Literal["mora", "jia", "b1"]

    def f(set_name: SET_NAMES = "mora") -> Path:
        if set_name == "jia":
            return here(remote, "2025-10-22_jia_seqlens", "datasets")
        elif set_name == "mora":
            return here(remote, "2025-11-21_mora_seqlens", "datasets")
        elif set_name == "b1":
            return here(remote, "2025-12-29_ast_b1", "datasets")
        raise ValueError(f"Set name must be one of {get_args(SET_NAMES)}")

    return f


@pytest.fixture
def rng(env):
    return np.random.default_rng(49274)


@pytest.fixture
def toy_dset(rng):
    def f(
        column_spec: dict = {},
        samples: Sequence | None | int = None,
        seq_level=False,
        x_size: int = 500,
        x_key: str = "x",
        n=1000,
    ) -> Dataset:
        if samples is None and not seq_level:
            samples = list(ascii_uppercase)
        elif samples is None and seq_level:
            samples = rng.choice(list(ascii_uppercase), n, replace=True)
        to_dset = {"sample": samples, x_key: torch.rand((len(samples), x_size))}
        for col, rvs in column_spec.items():
            if isinstance(rvs, Callable):
                to_dset[col] = rvs(rng, len(samples))
            else:
                to_dset[col] = rng.choice(rvs, len(samples), replace=True)
        return Dataset.from_dict(to_dset).with_format("torch")

    return f


@pytest.fixture
def random_linked_dset(tmp_path, rng, custom_cols: dict[str, Callable] | None = None):
    from mimesis import Food, Text

    txt = Text()
    food = Food()
    fruits = food._dataset["fruits"][:10]
    fruits_bin = food._dataset["fruits"][:2]
    color_bin = txt._dataset["color"][:2]
    dna = list("ATCG")

    def fn(n: int = 1000, dim: int = 10) -> LinkedDataset:
        annots = list(set(txt.words(20)))
        n_caches = len(list(tmp_path.glob("*cache*")))
        name = "cache" + f"_{n_caches}" if n_caches > 0 else "cache"
        cache_path = tmp_path / name
        cache_path.mkdir()

        def embed_fn(values) -> tuple[str, torch.Tensor, torch.Tensor]:
            for val in values:
                yield (val, torch.randn((8, dim)), torch.randn(rng.integers(9) + 1))

        dct = {
            "id": range(n),
            "word": txt.words(n),
            "c1": [txt.color() for _ in range(n)],
            "b1": rng.choice(fruits_bin, size=n, replace=True),
            "b2": rng.choice(color_bin, size=n, replace=True),
            "c2": rng.choice(fruits, size=n, replace=True),
            "s1": [
                "".join(
                    rng.choice(dna, size=rng.integers(low=20, high=300), replace=True)
                )
                for _ in range(n)
            ],
            "a1": [
                ";".join(rng.choice(annots, size=rng.integers(low=1, high=len(annots))))
                for _ in range(n)
            ],
        }
        if custom_cols is not None:
            for name, col_gen in custom_cols.items():
                dct[name] = col_gen(n)
        df = pl.DataFrame(dct)

        cache = EmbeddingCache(
            dir=cache_path, rng=rng, pooling=BasicPoolings.MEAN, save_mode="seqs"
        )
        cache.save(df["word"], embed_fn=embed_fn)
        dset = LinkedDataset(
            cache=cache, text_key="word", meta=df, level="seqs", x_key="x"
        )
        return dset

    return fn
