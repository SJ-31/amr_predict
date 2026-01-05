#!/usr/bin/env ipython


import pytest
from amr_predict.pooling import StaticPooler
from datasets import Dataset


@pytest.mark.parametrize(
    "kws",
    [
        {"method": "sum"},
        {"method": "mean"},
        {"method": "similarity"},
        {
            "method": "seq_subset",
            "priority": ["bar", "baz", "foo"],
            "subset_col": "ctrl",
        },
        {"method": "random"},
    ],
)
def test_alignment(kws, toy_dset):
    dset: Dataset = toy_dset(
        {"class": ["R", "S", "I"], "ctrl": ["foo", "bar", "baz", "bat", "bun"]},
        seq_level=True,
        x_key="embedding",
    )
    metadata = dset.to_polars().unique("sample")
    sp = StaticPooler(obs_keep=["ctrl", "class"], **kws)
    pooled = sp(dset).to_polars()
    joined = metadata.join(pooled, how="inner", on="sample")
    assert joined.height == metadata.height
    assert (joined["ctrl"] == joined["ctrl_right"]).all()
    assert (joined["class"] == joined["class_right"]).all()
