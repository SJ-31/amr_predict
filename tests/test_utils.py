#!/usr/bin/env ipython

from string import ascii_letters

import torch
from amr_predict.utils import EmbeddingCache
from loguru import logger

logger.enable("amr_predict")


def dummy_embed(texts) -> dict:
    ncol = 3
    mapping = dict(zip(ascii_letters, range(len(ascii_letters))))

    def embed(text):
        return torch.tensor([mapping[text[i]] for i in range(ncol)])

    return dict(zip(texts, [embed(t) for t in texts]))


def test_cache1(tmp_path):
    path = tmp_path / ".cache"
    path.mkdir()
    cache: EmbeddingCache = EmbeddingCache(path)
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
    cache.save(words, fn=dummy_embed, batch_size=3)
    assert "bridge" in cache
    assert "foo" not in cache
    assert len(cache) == len(words)
    assert len(cache.values()) == len(set(cache.values()))
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
