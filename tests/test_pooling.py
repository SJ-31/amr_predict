#!/usr/bin/env ipython

import numpy as np
import polars as pl
import pytest
from amr_predict import pooling
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
        {"method": "max"},
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
    tmp = sp(dset)
    embedding_size = dset["embedding"][:].shape[1]
    assert len(tmp["x"][:].shape) == 2
    assert tmp["x"][:].shape[1] == embedding_size
    pooled = tmp.to_polars()
    joined = metadata.join(pooled, how="inner", on="sample")
    assert joined.height == metadata.height
    assert (joined["ctrl"] == joined["ctrl_right"]).all()
    assert (joined["class"] == joined["class_right"]).all()
    pol = dset.to_polars()
    print(pol["sample"].value_counts())
    ref = next(iter(metadata["sample"].unique()))
    print(f"Reference sample: {ref}")
    ref_arr = np.array(pol.filter(pl.col("sample") == ref)["embedding"])
    test_arr = np.array(pooled.filter(pl.col("sample") == ref)["x"][0])
    if kws["method"] == "mean":
        mean_true = ref_arr.mean(axis=0)
        mean_test = test_arr
        assert (mean_true - mean_test) == pytest.approx(0, abs=1e-6)
    elif kws["method"] == "sum":
        assert (ref_arr.sum(axis=0) - test_arr) == pytest.approx(0, abs=1e-5)
