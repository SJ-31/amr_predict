#!/usr/bin/env ipython

import os

import numpy as np
import polars as pl
import pytest
from amr_predict.embedding import EmbeddingModels, ModelEmbedder
from amr_predict.enums import BasicPoolings
from loguru import logger
from numpy.random import Generator
from pyhere import here

logger.enable("amr_predict")

# [2026-04-23 Thu] TODO: run this test


@pytest.mark.parametrize(
    "model,seqtype,save_mode,lg",
    [
        (EmbeddingModels.seqLens_4096_512_46M_Mp, "nuc", "seqs", False),
        (EmbeddingModels.seqLens_4096_512_46M_Mp, "nuc", "tokens", False),
        (EmbeddingModels.seqLens_4096_512_46M_Mp, "nuc", "tokens", False),
        (EmbeddingModels.omniNA_66m, "nuc", "seqs", False),
        (EmbeddingModels.ntv3_8m_pre, "nuc", "seqs", False),
        (EmbeddingModels.esmc_600m, "aa", "seqs", False),
        (EmbeddingModels.esmc_600m, "aa", "tokens", True),
    ],
)
def test_embedding(tmp_path, model, seqtype, save_mode, lg):
    HF = here("data", "remote", "cache", "huggingface")
    os.environ["HF_HOME"] = str(HF)
    rng: Generator = np.random.default_rng()
    workdir = tmp_path / "cache"
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
        model=model,
        batch_size=10,
        workdir=workdir,
        save_mode=save_mode,
        save_proba=lg,
        pooling=BasicPoolings.MEAN,
        only_cache=True,
        hidden_layer=0,
        huggingface=None,
    )
    E.embed(dataset)
    print(E.cache.to_pl().collect())
