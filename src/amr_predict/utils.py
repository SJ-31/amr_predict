#!/usr/bin/env ipython
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import anndata as ad
import numpy as np
import polars as pl
import torch
import torch.utils.data as td
from Bio.SeqRecord import SeqRecord
from datasets import DatasetDict, Value
from datasets.arrow_dataset import Dataset
from datasets.load import load_from_disk
from torch import Tensor

# * Utility functions

TASK_TYPES: TypeAlias = Literal["classification", "regression"]


def iter_cols(x: Tensor | np.ndarray | tuple) -> Iterable:
    """Iterate over columnes of x"""
    if isinstance(x, Tensor):
        to_iter = torch.unbind(x, dim=1)
    elif isinstance(x, np.ndarray):
        to_iter = [x[:, i] for i in range(x.shape[1])]
    else:
        to_iter = iter(x)
    return to_iter


def join_within(
    left: pl.DataFrame,
    right: pl.DataFrame,
    initial_join: Sequence,
    start_col: str = "start",
    stop_col: str = "stop",
) -> pl.DataFrame:
    """Messy attempt at a left-based `within` join

    Parameters
    ----------
    left : pl.DataFrame
    right : pl.DataFrame
    initial_join : Sequence
        columns to initially join left, right by

    Returns
    -------
    DataFrame where entries of `right` match entries of `left`, and were originally contained
        within (`start_col`, `stop_col`) of `left`
        You can then join this column back into left using `uid`

    Notes
    -----

    """
    valid = []
    unwanted = set(initial_join) | {start_col, stop_col}
    wanted_cols = ["index"] + [c for c in right.columns if c not in unwanted]
    left = left.with_row_index()
    grouped = left.join(right, on=initial_join, how="left").group_by(initial_join)
    for _, g in grouped:
        # Accept where a sequence in `left` falls completely into a sequence on `right`
        # and vice versa
        filtered = g.filter(
            (  # right within
                (pl.col(f"{start_col}_right") >= pl.col(start_col))
                & (pl.col(f"{stop_col}_right") <= pl.col(stop_col))
            )
            | (  # left within
                (pl.col(f"{start_col}_right") <= pl.col(start_col))
                & (pl.col(f"{stop_col}_right") >= pl.col(stop_col))
            )
        )
        if not filtered.is_empty():
            valid.append(filtered)
    df = pl.concat(valid).select(wanted_cols).unique("index")
    left = left.join(df, on="index", how="left", maintain_order="left").drop("index")
    return left


def add_intergenic(
    record: SeqRecord,
    df: pl.DataFrame,
    start_col: str = "Start",
    stop_col: str = "Stop",
) -> pl.DataFrame:
    final_intergenic = len(record) - max(df[stop_col])
    df = df.with_columns(
        downstream_intergenic=np.concat(
            [df["Start"][1:] - df["Stop"][:-1], [final_intergenic]]
        )
    )
    first_intergenic = min(df[start_col])
    df = df.with_columns(
        upstream_intergenic=np.concat(
            [[first_intergenic], df["downstream_intergenic"][:-1]]
        )
    )
    return df


def read_tabular(file: Path | str, infer_schema_length=None, **kwargs) -> pl.DataFrame:
    file = file if isinstance(file, Path) else Path(file)
    sep = "\t" if file.suffix == ".tsv" else ","
    df: pl.DataFrame | None = pl.read_csv(
        file, separator=sep, infer_schema_length=infer_schema_length, **kwargs
    )
    return df


def load_as(
    dset_path,
    format: Literal["huggingface", "torch", "adata", "polars"] = "huggingface",
    columns: Sequence | None = None,
    x_key: str | None = None,
) -> Dataset | td.Dataset | DatasetDict | ad.AnnData | pl.DataFrame:
    """Load huggingface dataset and convert to pytorch tensors

    Parameters
    ----------
    columns : Sequence | None
        ordered sequence of columns to keep in the dataset
    """
    load_to = "torch" if format != "adata" else "numpy"
    dset = load_from_disk(dset_path).with_format(load_to)
    to_keep = columns or dset.column_names
    if format == "huggingface":
        return dset.select_columns(to_keep)
    elif format == "adata":
        x_key = x_key if x_key is not None else to_keep[0]
        x = dset[x_key][:]
        obs = dset.select_columns(to_keep).to_pandas()
        return ad.AnnData(X=x, obs=obs)
    elif format == "polars":
        return dset.to_polars().select(to_keep)
    else:
        feature_types = dset.features
        to_keep = [k for k in to_keep if feature_types[k] != Value("string")]
        return td.TensorDataset(*[dset[k][:] for k in to_keep])


def dataset2adata(dset: td.Dataset | Dataset, x_key: str = "embedding") -> ad.AnnData:
    import pandas as pd

    if isinstance(dset, td.Dataset):
        tensors = {f"v{i}": v for i, v in enumerate(dset[:])}
        x = tensors.pop("v0")
        if not isinstance(x, np.ndarray):
            x = x.numpy()
        obs = pd.DataFrame(tensors)
    else:
        x = dset[x_key][:].numpy()
        obs = dset.remove_columns(x_key).to_pandas()
    return ad.AnnData(X=x, obs=obs)


# * Classes


@dataclass
class ModuleConfig:
    def __init__(
        self,
        record_metrics: bool = True,
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
        scheduler_config: dict | None = None,
        cache: str | None | Sequence = None,
        record_norm: bool = False,
        dropout_p: float = 0.2,
        init_device: str = "cpu",
        n_classes_per_task: list[int] | None = None,
        task_type: TASK_TYPES = "regression",
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        task_weights : Tensor | Sequence | None
            task_weights
        log_norm : bool
            Whether to log the gradient norm
        cache : str | Sequence None
            Module attributes to cache e.g. for logging
        scaler : TorchScaler
            Fitted (ideally on entire train set) scaler that will apply transformation
            to each batch prior to training
        kwargs : Model-specific kwargs, stored in a dict
        targets : Tensor of targets that the model will observe during training
        """
        self.record_norm: bool = record_norm
        self.record: bool = record_metrics
        self._init_device: torch.device = torch.device(init_device)
        self.optimizer_fn: Callable | None = optimizer_fn
        self.n_classes_per_task: list[int] | None = n_classes_per_task
        self.scheduler_fn: Callable | None = scheduler_fn
        self.scheduler_config: dict | None = scheduler_config
        self.dropout_p: float = dropout_p
        self.cache: str | Sequence | None = cache
        self.task_type: TASK_TYPES = task_type
        self.kwargs: dict = kwargs

    @property
    def init_device(self) -> torch.device:
        return self._init_device

    @init_device.setter
    def init_device(self, value: str | torch.device):
        if isinstance(value, str):
            self._init_device = torch.device(value)
        else:
            self._init_device = value
