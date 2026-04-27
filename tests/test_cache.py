#!/usr/bin/env ipython

from pathlib import Path
from string import ascii_letters
from typing import Callable
from uuid import uuid4

import numpy as np
import polars as pl
import pytest
import torch
from amr_predict.cache import EmbeddingCache, LinkedDataset
from datasets import Dataset
from pyhere import here
from torch.utils.data import DataLoader


def dummy_embed(texts):
    ncol = 3
    mapping = dict(zip(ascii_letters, range(len(ascii_letters))))

    def embed(text):
        return torch.tensor([mapping[text[i]] for i in range(ncol)]).to(torch.float32)

    for t in texts:
        yield (
            t,
            torch.vstack([embed(t) for _ in range(torch.randint(2, 10, (1,)))]),
            torch.tensor([mapping[letter] + torch.rand(1).item() for letter in t]),
        )


@pytest.fixture
def rng():
    return np.random.default_rng()


@pytest.fixture
def make_default_cache(tmp_path, rng) -> Callable:
    def fn(with_random: bool = False, mode="both", save_proba: bool = False):
        path: Path = tmp_path / str(uuid4()) / ".cache"
        path.mkdir(parents=True)
        cache: EmbeddingCache = EmbeddingCache(
            path, save_interval=1, save_mode=mode, save_proba=save_proba
        )
        words = [
            "forest",
            "crane",
            "marble",
            "silver",
            "tiger",
            "planet",
            "shadow",
            "bridge",
            "coral",
            "ember",
            "novel",
            "quartz",
            "raven",
            "flame",
            "harbor",
        ]
        if with_random:
            new_words = [
                word + "".join(rng.choice(list(ascii_letters), 5)) for word in words
            ]
            words.extend(new_words)
        cache.save(words, embed_fn=dummy_embed, batch_size=2)
        return cache, words

    return fn


def test_cache1_only_tk(make_default_cache):
    cache: EmbeddingCache
    cache, words = make_default_cache(mode="tokens")
    print(cache.to_pl().collect())


def test_cache1(make_default_cache):
    cache: EmbeddingCache
    cache, words = make_default_cache()
    assert "bridge" in cache
    assert "foo" not in cache
    assert len(cache) == len(words)
    assert cache.to_pl().collect().height == len(words)
    print(cache.to_pl(as_array=True))
    path = cache.dir

    assert (cache["forest"] == torch.tensor([5, 14, 17])).all()
    assert (cache["harbor"] == torch.tensor([7, 0, 17])).all()

    c2: EmbeddingCache = EmbeddingCache(path)
    assert c2.keys() == cache.keys()
    assert (c2["forest"] == torch.tensor([5, 14, 17])).all()
    assert (c2["novel"] == torch.tensor([13, 14, 21])).all()
    assert (c2["harbor"] == torch.tensor([7, 0, 17])).all()

    words2 = [
        "cascade",
        "meadow",
        "pillar",
        "linen",
        "throne",
        "spice",
        "amber",
        "whisk",
    ]
    cache.save(words2, embed_fn=dummy_embed, batch_size=5)
    assert (cache["harbor"] == torch.tensor([7, 0, 17])).all()
    assert (cache["linen"] == torch.tensor([11, 8, 13])).all()
    old_len = len(cache)
    dupes = [
        "cascade",
        "meadow",
        "throne",
        "spice",
        "amber",
        "whisk",
    ]
    cache.save(dupes, embed_fn=dummy_embed, batch_size=5)
    assert old_len == len(cache)
    print(cache.retrieve(pl.Series(["cascade", "meadow", "spice"]), level="seqs"))


def test_cache2(make_default_cache):
    cache: EmbeddingCache
    cache, words = make_default_cache()
    cache.rewrite(n_rows=3)
    print(list(cache.dir.iterdir()))
    prop = 0.3
    cache.rewrite(n_rows=3, token_prop=prop)
    df = cache.to_pl().collect()
    null_count = df.filter(pl.col("token").is_not_null()).height
    print(null_count)
    print(df.height * prop)
    assert pytest.approx(null_count, abs=3) == df.height * prop


@pytest.mark.parametrize("level", ["tokens", "seqs"])
def test_cache2df(level, make_default_cache):
    cache: EmbeddingCache
    cache, words = make_default_cache(save_proba=True)
    dset: LinkedDataset = cache.to_dataset(
        df=pl.DataFrame({"words": words}), key_col="words", level=level
    )
    df = dset.to_pl()
    adata = dset.to_anndata()
    if level == "seqs":
        assert df.height == len(words)
        assert adata.shape[0] == len(words)


def test_cache_combine(make_default_cache, tmp_path):
    cache, words = make_default_cache()
    cache2, words2 = make_default_cache(True)
    combined = EmbeddingCache.combine([cache, cache2], new_path=tmp_path)
    df: pl.DataFrame = combined.to_pl().collect()
    assert (set(words) | set(words2)) == set(df["key"])


def test_dataset(make_default_cache):
    cache: EmbeddingCache
    cache, words = make_default_cache()
    df = pl.DataFrame({"key": words[:5], "sample": range(5)})
    ds = LinkedDataset(meta=df, cache=cache, text_key="key", x_key="x")
    loader = DataLoader(ds, batch_size=3)
    assert ds[:2]["x"].shape[0] == 2
    assert next(iter(loader))["x"].shape[0] == 3
    print(cache.to_pl(as_array=True).collect())
    d2 = LinkedDataset(meta=df, cache=cache, text_key="key", x_key="x", level="tokens")
    print(d2[:])
    print(d2.shape)
    assert len(d2) == d2[:]["x"].shape[0]
    assert ds.meta.shape[0] == ds["x"].shape[0]
    assert ds.meta.shape[0] == ds[:]["x"].shape[0]
