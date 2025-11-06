#!/usr/bin/env ipython

from pathlib import Path

import polars as pl
import polars.selectors as cs
import yaml
from amr_predict.utils import discretize_resistance, read_tabular
from pyhere import here

mpath: Path = here("data", "meta")

with open(mpath.joinpath("specification.yaml"), "rb") as f:
    spec = yaml.safe_load(f)

dsuffix: str = spec["discretize_suffix"]

metas: list[pl.DataFrame] = []
for name, dset_spec in spec["metadata"].items():
    df: pl.DataFrame = read_tabular(mpath.joinpath(dset_spec["path"]))
    if scol := dset_spec.get("sample_col"):
        df = df.rename({scol: spec["sample_col"]})
    if make_discrete := dset_spec.get("discretize"):
        df = discretize_resistance(df, suffix=dsuffix, **make_discrete)
    if class_cols := dset_spec.get("add_class_suffix"):
        df = df.rename({c: f"{c}_{dsuffix}" for c in class_cols})
    if to_drop := dset_spec.get("to_drop"):
        df = df.drop(to_drop)
    metas.append(df.with_columns(pl.lit(name).alias("dataset")))

meta: pl.DataFrame = pl.concat(metas, how="diagonal_relaxed").select(
    "sample", "dataset", cs.exclude(["sample", "dataset"])
)
