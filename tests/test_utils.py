#!/usr/bin/env ipython

from string import ascii_letters

import numpy as np
import polars as pl
import pytest
import torch
from amr_predict.utils import EmbeddingCache
from loguru import logger

logger.enable("amr_predict")

rng = np.random.default_rng()

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

dummy_df = pl.DataFrame({"a": list(ascii_letters), "b": torch.rand(len(ascii_letters))})

pl.LazyFrame(
    {"a": "a", "b": [torch.rand(9)]},
    schema={"a": pl.String, "b": pl.Array(pl.Float64, 9)},
).collect()

dummy_df.with_columns(
    pl.when(pl.Series(rng.choice([True, False], p=[0.3, 0.7], size=dummy_df.height)))
    .then(pl.lit(None))
    .otherwise(pl.col("a"))
    .alias("a")
)["a"].value_counts()

dummy_df.filter(rng.choice([True, False], p=[0.3, 0.7], size=dummy_df.height))


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
def default_cache(tmp_path) -> tuple[EmbeddingCache, list]:
    path = tmp_path / ".cache"
    path.mkdir()
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
    cache.save(words, fn=dummy_embed, batch_size=2)
    return cache, words


def test_cache1(default_cache):
    cache: EmbeddingCache
    cache, words = default_cache
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


def test_cache2(default_cache):
    cache: EmbeddingCache
    cache, words = default_cache
    cache.rewrite(n_rows=3)
    print(list(cache._dir.iterdir()))
    # assert len(list(cache._dir.iterdir())) == 2
    prop = 0.3
    cache.rewrite(n_rows=3, token_prop=prop)
    df = cache.pl().collect()
    null_count = df.filter(pl.col("token").is_not_null()).height
    print(null_count)
    print(df.height * prop)
    assert pytest.approx(null_count, abs=2) == df.height * prop
