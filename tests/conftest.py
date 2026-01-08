#!/usr/bin/env ipython

from collections.abc import Sequence
from string import ascii_uppercase

import numpy as np
import pytest
import tomllib
import torch
import yaml
from datasets import Dataset
from pyhere import here


@pytest.fixture
def env():
    with open(here("tests", "env.toml"), "rb") as f:
        env: dict = tomllib.load(f)

    with open(here("snakemake", "env.yaml"), "rb") as f:
        env.update(yaml.safe_load(f))
    return env


@pytest.fixture
def keys(env):
    return (
        env["pool_embeddings"]["key"],
        env["pool_embeddings"]["sample_key"],
    )


@pytest.fixture
def rng(env):
    return np.random.default_rng(env["rng"])


@pytest.fixture
def toy_dset(rng):
    def f(
        column_spec: dict = {},
        samples: Sequence | None = None,
        seq_level=False,
        x_size: int = 500,
        x_key: str = "x",
        n=1000,
    ) -> Dataset:
        if samples is None and not seq_level:
            samples = list(ascii_uppercase)
        elif samples is None and seq_level:
            samples = rng.choice(list(ascii_uppercase), n, replace=True)
        to_dset = {"sample": samples, x_key: torch.rand(len(samples), x_size)}
        for col, choices in column_spec.items():
            to_dset[col] = rng.choice(choices, len(samples), replace=True)
        return Dataset.from_dict(to_dset).with_format("torch")

    return f
