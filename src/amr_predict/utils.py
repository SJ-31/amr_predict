#!/usr/bin/env ipython
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import reduce
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
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import DataLoader

# * Utility functions

TASK_TYPES: TypeAlias = Literal["classification", "regression"]


def encode_strs(
    data: Dataset | ad.AnnData, task_names: tuple
) -> tuple[Dataset | ad.AnnData, dict[str, LabelEncoder]]:
    encoders = {}
    for task in task_names:
        encoder = LabelEncoder()
        if isinstance(data, Dataset):
            task_vec = data[task][:]
            data = data.remove_columns(task).add_column(
                task, encoder.fit_transform(task_vec)
            )
        else:
            task_vec = data.obs[task]
            data.obs.loc[:, task] = encoder.fit_transform(task_vec)
        encoders[task] = encoder
    return data, encoders


def data_spec(
    X: torch.Tensor | np.ndarray | Dataset,
    y: torch.Tensor | np.ndarray | None | pl.DataFrame | Sequence = None,
    x_key: str | None = None,
) -> tuple[int, tuple[int, ...]]:
    """Return a tuple of (n_features, n_classes) for the given dataset
    If multitask, the second element is a tuple of length n_tasks
    """
    if isinstance(X, Dataset) and x_key is None:
        raise ValueError("x_key must be provided if given a dataset")
    if isinstance(X, Dataset):
        X = X[x_key][:]
    if isinstance(y, pl.DataFrame):
        return X.shape[1], tuple([len(y[s].unique()) for s in y])
    elif isinstance(y, np.ndarray) or isinstance(y, torch.Tensor) and len(y.shape) > 1:
        return X.shape[1], tuple([y[:, i].unique().shape[0] for i in range(y.shape[1])])
    return X.shape[1], tuple([len(set(y))])


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


def discretize_resistance(
    dataset: Dataset,
    cols: Sequence,
    susceptible: float = 0.3,
    resistant: float = 0.8,
    suffix: str = "",
) -> Dataset:
    """Discretize continuous resistance scores
    e.g. AST into one of three categories, using quantiles

    Parameters
    ----------
    susceptible : float
        values less than this quantile are categorized as "susceptible"
    resistant : float
        values greater than this are categorized as "resistant"
    suffix : str
        create new columns in `dataset` with this suffix. Otherwise, replace columns
    cols : Sequence
        columns to consider
    """
    df: pl.DataFrame = dataset.to_polars().select(cols)
    exprs = (
        pl.when(pl.col(col) < pl.col(col).quantile(susceptible))
        .then(pl.lit("susceptible"))
        .when(pl.col(col) > pl.col(col).quantile(resistant))
        .then(pl.lit("resistant"))
        .otherwise(pl.lit("intermediate"))
        .alias(col)
        for col in cols
    )
    df = df.with_columns(*exprs)
    if not suffix:
        dataset = dataset.remove_columns(cols)
    for col in cols:
        n = f"{col}_{suffix}" if suffix else col
        dataset = dataset.add_column(n, df[col])
    return dataset


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
        file,
        separator=sep,
        infer_schema_length=infer_schema_length,
        null_values="NA",
        **kwargs,
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
        to_keep.remove(x_key)
        obs = dset.remove_columns(x_key).select_columns(to_keep).to_pandas()
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


def vecdist(
    x: np.ndarray, y: np.ndarray, metric: Literal["cosine", "euclidean", "manhattan"]
) -> np.ndarray:
    """Element-wise distance calculation between rows of 2d matrices x, y"""
    if len(x.shape) > 2 or len(y.shape) > 2:
        raise ValueError("x and y must both be 2d")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y have a differing number of elements!")
    if metric == "cosine":
        nx = np.linalg.vector_norm(x, axis=1, ord=2)
        ny = np.linalg.vector_norm(y, axis=1, ord=2)
        return 1 - (np.vecdot(x, y) / (nx * ny))
    elif metric == "euclidean":
        return np.sqrt(np.sum((x - y) ** 2, axis=1))
    elif metric == "manhattan":
        return np.sum(np.abs((x - y)), axis=1)


def train_test_from_dict(df: pl.DataFrame, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Generate train, test splits from a dictionary and metadata dataframe

    Parameters
    ----------
    df : DataFrame
        dataframe where rows represent samples and columns are observation
    spec : dict
        Keys of `spec` refer to columns of `df`, and values are a dictionary that
            specify how to generate a boolean mask from different values of `spec`
        The keys of this dictionary are specific values of the column, and values are
            match types.

        Possible match types are CONTAINS, EXACT, NOT, CONTAINS_ANY

        Match types are interpreted as being for the test set
            e.g. "myvar": {"foo": EXACT} generates a mask where
            all samples with myvar == "foo" are placed in the test set

        The matches will be merged with a boolean OR operation

    Returns
    -------
    Tuple of boolean masks for train, test splits

    Notes
    ------
    CONTAINS and CONTAINS_ANY matches only work for string columns
    """
    test_masks = []
    for obs, val_dct in spec.items():
        for value, match_type in val_dct.items():
            match_type: Literal["EXACT", "NOT", "CONTAINS", "CONTAINS_ANY"]
            if match_type == "EXACT":
                test_masks.append(df[obs] == value)
            elif match_type == "CONTAINS":
                test_masks.append(df[obs].str.contains(value))
            elif match_type == "NOT":
                test_masks.append(df[obs] != value)
            elif match_type == "CONTAINS_ANY":
                test_masks.append(df[obs].str.contains_any(value))
            else:
                raise ValueError(f"`{match_type}` is an invalid match type!")
    test_mask: np.ndarray = reduce(lambda x, y: x | y, test_masks).to_numpy()
    return ~test_mask, test_mask


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
        n_tasks: int = 1,
        n_classes: list[int] | None = None,
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
        task_names : Sequence
            For supervised models, a sequence of task keys with which to access the target
            variables from the dataset during training
        targets : Tensor of targets that the model will observe during training
        """
        self.n_tasks: int = n_tasks
        self.record_norm: bool = record_norm
        self.record: bool = record_metrics
        self._init_device: torch.device = torch.device(init_device)
        self.optimizer_fn: Callable | None = optimizer_fn
        self.n_classes: list[int] | None = n_classes
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
