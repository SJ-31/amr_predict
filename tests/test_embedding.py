#!/usr/bin/env ipython

import numpy as np
import polars as pl
import pytest
from amr_predict.embedding import EmbeddingModels, ModelEmbedder
from loguru import logger
from numpy.random import Generator
from pyhere import here

HF = here("data", "remote", "cache", "huggingface")

logger.enable("amr_predict")

# [2026-04-23 Thu] TODO: run this test


@pytest.mark.parametrize(
    "model,seqtype,save_mode,lg",
    [
        (EmbeddingModels.seqLens_4096_512_46M_Mp, "nuc", "seqs", False),
        (EmbeddingModels.seqLens_4096_512_46M_Mp, "nuc", "tokens", False),
        (EmbeddingModels.seqLens_4096_512_46M_Mp, "nuc", "tokens", False),
        (EmbeddingModels.esmc_600m, "aa", "seqs", False),
        (EmbeddingModels.esmc_600m, "aa", "tokens", True),
    ],
)
def test_embedding(tmp_path, model, seqtype, save_mode, lg):
    workdir = tmp_path / "cache"
    rng: Generator = np.random.default_rng()
    choices = list("ATCG" if seqtype == "nuc" else "FRILSPEGYNA")
    n = 100
    dataset = pl.DataFrame(
        {
            "sample": range(n),
            "sequence": [
                "".join(rng.choice(choices, rng.integers(10, 100), replace=True))
                for _ in range(n)
            ],
        }
    )
    E = ModelEmbedder.new(
        model,
        batch_size=10,
        workdir=workdir,
        save_mode=save_mode,
        save_proba=lg,
        only_cache=True,
        hidden_layer=0,
        huggingface=str(HF),
    )
    E.embed(dataset)
    print(E.cache.to_pl().collect())
