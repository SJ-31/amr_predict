#!/usr/bin/env ipython

from pathlib import Path
from string import ascii_letters
from typing import Callable
from uuid import uuid4

import numpy as np
import polars as pl
import pytest
import torch
from amr_predict.utils import EmbeddingCache, LinkedDataset
from loguru import logger
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


def dummy_embed(texts):
    ncol = 3
    mapping = dict(zip(ascii_letters, range(len(ascii_letters))))

    def embed(text):
        return torch.tensor([mapping[text[i]] for i in range(ncol)])

    for t in texts:
        yield (
            t,
            embed(t),
            torch.vstack([embed(t) for _ in range(torch.randint(2, 10, (1,)))]),
        )
        # yield {
        #     "key": t,
        #     "seq": embed(t),
        #     "token": torch.vstack(
        #         [embed(t) for _ in range(torch.randint(2, 10, (1,)))]
        #     ),
        # }


@pytest.fixture
def make_default_cache(tmp_path, rng) -> Callable:
    def fn(with_random: bool = False):
        path: Path = tmp_path / str(uuid4()) / ".cache"
        path.mkdir(parents=True)
        cache: EmbeddingCache = EmbeddingCache(path, save_interval=1)
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
        cache.save(words, fn=dummy_embed, batch_size=2)
        return cache, words

    return fn


def test_cache1(make_default_cache):
    cache: EmbeddingCache
    cache, words = make_default_cache()
    assert "bridge" in cache
    assert "foo" not in cache
    assert len(cache) == len(words)
    assert cache.pl().collect().height == len(words)
    print(cache.pl(as_array=True))
    path = cache._dir

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
    cache.save(words2, fn=dummy_embed, batch_size=5)
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
    cache.save(dupes, fn=dummy_embed, batch_size=5)
    assert old_len == len(cache)
    print(cache.retrieve(["cascade", "meadow", "spice"]))


def test_cache2(make_default_cache):
    cache: EmbeddingCache
    cache, words = make_default_cache()
    cache.rewrite(n_rows=3)
    print(list(cache._dir.iterdir()))
    # assert len(list(cache._dir.iterdir())) == 2
    prop = 0.3
    cache.rewrite(n_rows=3, token_prop=prop)
    df = cache.pl().collect()
    null_count = df.filter(pl.col("token").is_not_null()).height
    print(null_count)
    print(df.height * prop)
    assert pytest.approx(null_count, abs=3) == df.height * prop


def test_cache_combine(make_default_cache, tmp_path):
    cache, words = make_default_cache()
    cache2, words2 = make_default_cache(True)
    combined = EmbeddingCache.combine([cache, cache2], new_path=tmp_path)
    df: pl.DataFrame = combined.pl().collect()
    assert (set(words) | set(words2)) == set(df["key"])


def test_dataset(make_default_cache):
    cache: EmbeddingCache
    cache, words = make_default_cache()
    df = pl.DataFrame({"key": words[:5], "sample": range(5)})
    ds = LinkedDataset(meta=df, cache=cache, text_key="key", x_key="x")
    loader = DataLoader(ds, batch_size=3)
    assert ds[:2]["x"].shape[0] == 2
    assert next(iter(loader))["x"].shape[0] == 3
    print(cache.pl(as_array=True).collect())
    d2 = LinkedDataset(
        meta=df, cache=cache, text_key="key", x_key="x", token_level=True
    )
    print(d2[:2])
    print(d2.shape)
    assert ds.meta.shape[0] == ds["x"].shape[0]
    assert ds.meta.shape[0] == ds[:]["x"].shape[0]


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
