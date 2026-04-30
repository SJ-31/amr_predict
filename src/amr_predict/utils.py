#!/usr/bin/env ipython
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import batched
from pathlib import Path
from typing import Any, Literal, TypeAlias

import anndata as ad
import jaxtyping
import numpy as np
import pandas as pd
import polars as pl
import skbio as sb
import torch
import torch.utils.data as td
from beartype import beartype
from datasets import DatasetDict, Features, Value, concatenate_datasets
from datasets.arrow_dataset import Dataset
from datasets.load import load_from_disk
from loguru import logger
from numpy.random import Generator
from skbio import DNA
from torch import Tensor

CACHE_OPTIONS: TypeAlias = Literal["train_loss", "val_acc", "val_loss", "train_acc"]
PP_METHODS: TypeAlias = Literal["variance"]


logger.disable("amr_predict")


# * Utility functions

TASK_TYPES: TypeAlias = Literal["classification", "regression", "reconstruction"]


def expand_annotations(col: pl.Series | pd.Series, split: str = ";") -> np.ndarray:
    """
    Expand a series of string annotations e.g. ["anno1;anno2", "anno35;anno9;anno10", ...]
    into a binary matrix of samples x annotations
    """
    if isinstance(col, pd.Series):
        col = pl.from_pandas(col)
    return (
        col.str.split(split)
        .to_frame()
        .with_row_index()
        .explode(col.name)
        .with_columns(val=pl.lit(1))
        .pivot(on=col.name, index="index", values="val", aggregate_function="first")
        .fill_null(0)
        .drop("index")
        .to_numpy()
    )


@beartype
def resample_pairs(
    x: jaxtyping.Shaped[Any, "a"] | Sequence,
    n: int = 1000,
    rng: int | Generator | None = None,
    only_indices: bool = True,
) -> jaxtyping.Integer[np.ndarray, "a b"] | list[tuple]:
    """
    Generate `n` random pairs from `x`

    Parameters
    ----------
    only_indices : bool
        If true, return an array of indices with shape (n, 2) containing indices for
        pairs in `x`. Otherwise, return a list of tuples containing paired elements of x
    """
    rng = rng if rng is not None else np.random.default_rng(rng)
    seen: set = set()
    count: int = len(x)
    pair_count: int = 0
    repeat_allowed = n >= count**2
    acc = []
    while pair_count < n:
        to_add = []
        first_half = rng.choice(range(count), size=count // 2)
        sec_half = rng.permuted(list(set(range(count)) - set(first_half)))[
            : len(first_half)
        ]
        together = np.hstack([first_half.reshape(-1, 1), sec_half.reshape(-1, 1)])
        for i, (f, s) in enumerate(zip(first_half, sec_half)):
            pair = (f, s)
            if pair not in seen or repeat_allowed:
                seen.add(pair)
                pair_count += 1
                to_add.append(i)
                if not only_indices:
                    acc.append((x[f], x[s]))
            if pair_count >= n:
                break
        if only_indices:
            acc.append(together[to_add, :])
    if only_indices:
        return np.vstack(acc)
    return acc


def sample_pairs_by_col(
    df: pd.DataFrame,
    var: str,
    n_pairs_per: int = 20,
    id_col: str | None = None,
    rng: int | None = None,
    replace: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Helper function to return randomly-sampled matrices of related and unrelated pairs

    Parameters
    ----------
    var : str
        Variable determining whether samples are related
    n_pairs_per : int
        The number of pairs per unique value of df[`var`] to add to the pair lists
    Returns
    -------
    A tuple of (related pairs, unrelated pairs). Each element is a two-column matrix,
        with nrows ~ len(df[var].unique()) * n_pairs_per

    Notes
    -----
    The values of the lists are numeric sample indices, for compatibility
    """
    if id_col is None:
        id_col = "idx"
        df = df.assign(idx=np.arange(df.shape[0]))
    related_pairs = []
    unrelated_pairs = []
    n_to_sample = n_pairs_per * 2
    for group, row in df.groupby(var).agg({id_col: lambda x: list(x)}).iterrows():
        grouped_idx = row[id_col]
        if len(grouped_idx) <= n_to_sample and not replace:
            continue
        # Shuffle ids in the same group, and pair up the first (n_pairs_per * 2) ids
        # to be the related pairs
        choices = pd.Series(grouped_idx).sample(frac=1, random_state=rng)
        c1 = choices.head(n=n_pairs_per * 2)
        related_pairs.extend([np.array(p) for p in batched(c1, 2)])

        # the bottom half of the shuffled sample become the unrelated pairs
        c2 = choices.tail(n=n_pairs_per)
        from_other_group: pd.DataFrame = (
            df.loc[df[var] != group, :][id_col]
            .sample(frac=1, random_state=rng)
            .head(n_pairs_per)
        )
        unrelated_pairs.extend([np.array(p) for p in zip(c2, from_other_group)])
    return np.array(related_pairs), np.array(unrelated_pairs)


def deduplicate(dset: Dataset, key: str, num_proc: int = 64) -> Dataset:
    # https://huggingface.co/datasets/Finnish-NLP/mc4_fi_cleaned/blob/main/deduplicate.py
    uniques: set = set(dset[key][:])

    def check_unique(x):
        if x[key] in uniques:
            uniques.remove(x[key])
            return True
        return False

    return dset.filter(lambda x: check_unique(x), num_proc=num_proc)


def debug_tensor_vals(tensor: Tensor, name: str):
    logger.debug(
        """
    Checking values of tensor `{}`
       contains nan: {}
       is all nan: {}
       nan_prop: {}
       contains inf: {}
       is all inf: {}
    """,
        name,
        tensor.isnan().any(),
        tensor.isnan().all(),
        tensor.isnan().sum() / tensor.numel(),
        tensor.isinf().any(),
        tensor.isinf().all(),
    )


def plot_params(key: str, cfg):
    top = cfg["report_plots"]
    lookup = top.get(key)
    if not lookup:
        return top["default"]
    return lookup


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
    if isinstance(X, Dataset) and y is not None and isinstance(y[0], str):
        return X[x_key][:].shape[1], tuple([len(np.unique(X[cls][:])) for cls in y])
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
    grouped = left.join(
        right, on=initial_join, how="left", maintain_order="left"
    ).group_by(initial_join)
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
    dataset: Dataset | pl.DataFrame,
    cols: Sequence,
    susceptible: float = 0.3,
    resistant: float = 0.8,
    suffix: str = "",
) -> Dataset | pl.DataFrame:
    """Discretize continuous resistance scores
    e.g. AST into one of three categories, using quantiles
    WARNING: it's AST are already discrete, and categorizing them into resistant,
    susceptible is species- and antimicrobial-specific

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
    was_df = isinstance(dataset, pl.DataFrame)
    df: pl.DataFrame = dataset.to_polars().select(cols) if not was_df else dataset
    exprs = (
        pl.when(pl.col(col) < pl.col(col).quantile(susceptible))
        .then(pl.lit("susceptible"))
        .when(pl.col(col) > pl.col(col).quantile(resistant))
        .then(pl.lit("resistant"))
        .otherwise(pl.lit("intermediate"))
        .alias(col)
        for col in cols
    )
    tmp = df.with_columns(*exprs)
    if not suffix and not was_df:
        dataset = dataset.remove_columns(cols)
    elif not suffix:
        return tmp
    if not was_df:
        for col in cols:
            n = f"{col}_{suffix}" if suffix else col
            dataset = dataset.add_column(n, tmp[col])
        return dataset
    return pl.concat(
        [df, tmp.select(cols).rename({c: f"{c}_{suffix}" for c in cols})],
        how="horizontal",
    )


def add_intergenic(
    record: sb.Sequence,
    df: pl.DataFrame,
    start_col: str = "Start",
    stop_col: str = "Stop",
) -> pl.DataFrame:
    """
    Add intergenic lengths to the feature dataframe
    E.g. if feature A has a start, end = 10, 500 and
        feature B has a start, end = 620, 1000, this function adds two columns

        feature A upstream_intergenic, downstream_intergenic = 10, 120
        feature B upstream_intergenic, downstream_intergenic = 120, ...

    Parameters
    ----------
    record : Sequence
        skbio sequence describing a CONTIG (not a gene sequence)
    start_col : str
        Column denoting feature start
    stop_col : str
        Column denoting feature stop
    """
    final_intergenic = len(record) - max(
        df[stop_col]
    )  # Remaining sequence with no features
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


def split_features(
    df: pl.DataFrame,
    max_length: int | None,
    start_col: str = "Start",
    end_col: str = "Stop",
    indicate_ends: bool = True,
    prefix: str = "chunk",
    drop_cols: bool = True,
) -> pl.DataFrame:
    """Split features into chunks if they exceed `max_length`

    Parameters
    ----------
    max_length : int | None
        Maximum feature length, and the length of the resulting chunks if features
        exceed this. If None, just the actual feature length
    indicate_ends : bool
        If true, add two boolean columns `is_5prime` and `is_3prime`
        indicating whether a feature chunk is at its 5' and/or 3' end respectively
        Both these columns are true if the feature was not split (shorter or equal to max length)
    prefix : str
        Prefix of the new columns indicating the start, end of the chunk.
        Format is {prefix}_`start_col`, {prefix}_`end_col`

    Returns
    -------
    An exploded dataframe with multiple rows per feature
    """
    to_remove = ("length", "n_chunks", "remaining")
    df = df.with_columns(length=pl.col(end_col) - pl.col(start_col))
    if max_length is not None:
        df = df.with_columns(
            n_chunks=pl.col("length") // max_length,
            remaining=pl.col("length") % max_length,
        )
    else:
        df = df.with_columns(n_chunks=0, remaining=0)

    def map_helper(x, col: Literal["start", "stop"] = "start"):
        nc = x["n_chunks"]
        start, stop = x[start_col], x[end_col]
        remaining = x["remaining"]
        has_remaining = remaining != 0 or remaining != x["length"]
        extend_to = max_length or x["length"]
        if col == "start":
            if not nc:
                return [start]
            val = [start + extend_to * i for i in range(nc)]
            if has_remaining:
                val.append(start + extend_to * nc)
            return val
        elif col == "stop":
            if not nc:
                return [stop]
            val = [start + extend_to * (i + 1) for i in range(nc)]
            if has_remaining:
                val.append(stop)
            return val

    c_start = f"{prefix}_{start_col}"
    c_end = f"{prefix}_{end_col}"
    into_struct = ["n_chunks", "remaining", "length", start_col, end_col]
    df = df.with_columns(
        pl.Series(range(len(df))).alias("feature_id"),
        pl.struct(into_struct)
        .map_elements(lambda x: map_helper(x, "start"), return_dtype=pl.List(pl.Int64))
        .alias(c_start),
        pl.struct(into_struct)
        .map_elements(lambda x: map_helper(x, "stop"), return_dtype=pl.List(pl.Int64))
        .alias(c_end),
    ).explode([c_start, c_end])
    if indicate_ends:
        df = df.with_columns(
            pl.when(pl.col(start_col) == pl.col(c_start))
            .then(True)
            .otherwise(False)
            .alias("is_5prime"),
            pl.when(pl.col(end_col) == pl.col(c_end))
            .then(True)
            .otherwise(False)
            .alias("is_3prime"),
        )
    if drop_cols:
        return df.drop(to_remove)
    return df


def read_tabular(file: Path | str, infer_schema_length=None, **kwargs) -> pl.DataFrame:
    file = file if isinstance(file, Path) else Path(file)
    if file.suffix == ".parquet":
        return pl.read_parquet(file, **kwargs)
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
        x: np.ndarray = x.numpy() if isinstance(x, Tensor) else x
        x = np.nan_to_num(x, nan=0.0)
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
            match_type: Literal["EXACT", "NOT", "CONTAINS", "CONTAINS_ANY", "LT", "GT"]
            if match_type == "EXACT":
                test_masks.append(df[obs] == value)
            elif match_type == "LT":
                value = float(value) if isinstance(value, str) else value
                test_masks.append(df[obs] < value)
            elif match_type == "GT":
                value = float(value) if isinstance(value, str) else value
                test_masks.append(df[obs] > value)
            elif match_type == "CONTAINS":
                test_masks.append(df[obs].str.contains(value))
            elif match_type == "NOT":
                test_masks.append(df[obs] != value)
            elif match_type == "CONTAINS_ANY":
                test_masks.append(df[obs].str.contains_any(value))
            else:
                raise ValueError(f"`{match_type}` is an invalid match type!")
    test_mask: np.ndarray = (
        reduce(lambda x, y: x | y, test_masks).to_numpy().astype(np.bool)
    )
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
        cache: tuple[CACHE_OPTIONS] | CACHE_OPTIONS | None = None,
        record_norm: bool = False,
        dropout_p: float = 0.2,
        init_device: str = "cpu",
        seed: int = 3110,
        n_tasks: int = 1,
        n_classes: tuple[int] = (1,),
        task_type: TASK_TYPES = "regression",
        task_names: tuple | None = None,
        task_weights: Tensor | None = None,
        **kws,
    ) -> None:
        self.n_tasks: int = n_tasks
        self.record_norm: bool = record_norm
        self.record: bool = record_metrics
        self._init_device: torch.device = torch.device(init_device)
        self.task_names: tuple = task_names or ()
        self.optimizer_fn: Callable | None = optimizer_fn
        self.n_classes: tuple[int] = n_classes
        self.scheduler_fn: Callable | None = scheduler_fn
        self.scheduler_config: dict | None = scheduler_config
        self.dropout_p: float = dropout_p
        self.cache: CACHE_OPTIONS | tuple[CACHE_OPTIONS] | None = cache
        self.task_type: TASK_TYPES = task_type
        self.task_weights: Tensor | None = task_weights
        self.seed = kws.get("seed", seed)
        self.kws: dict = kws

    def get(self, key: str, default):
        if key in self:
            return self[key]
        return default

    def __repr__(self) -> str:
        tmp = {}
        attrs = [k for k in dir(self) if not k.startswith("__")]
        for key in attrs:
            if key not in {"kws", "get"} and key != "_init_device":
                tmp[key] = getattr(self, key)
        tmp.update(self.kws)
        return repr(tmp)

    def __contains__(self, key):
        return key in dir(self) or key in self.kws

    def __getitem__(self, key: str):
        if key in dir(self):
            return getattr(self, key)
        return self.kws[key]

    @property
    def init_device(self) -> torch.device:
        return self._init_device

    @init_device.setter
    def init_device(self, value: str | torch.device):
        if isinstance(value, str):
            self._init_device = torch.device(value)
        else:
            self._init_device = value


# ** Preprocessing


class Preprocessor:
    """
    Class to preprocess datasets prior to training e.g. apply data transformation
    or filter features
    Intended to be used with Evaluator class, which will call the `fit` method only
    on training datasets
    """

    def __init__(
        self,
        method: PP_METHODS,
        x_key: str = "x",
        feature_file: Path | None = None,
        read_idx: bool = False,
        **kws,
    ) -> None:
        self.method: PP_METHODS = method
        self.feature_file: Path | None = feature_file
        self.feature_idx: Sequence | None = None
        self.x_key: str = x_key
        if read_idx and feature_file is None:
            raise ValueError("`read_idx` was passed without `feature_file`")
        elif read_idx:
            self._read_idx()
        self.kws: dict = kws or {}

    def _filter_idx(self, batch):
        batch[self.x_key] = batch[self.x_key][:, self.feature_idx]
        return batch

    def transform(self, dataset: Dataset) -> Dataset:
        if self.method in {"variance"}:
            filtered = dataset.map(
                self._filter_idx, batched=True, batch_size=dataset.shape[0]
            )
            return filtered
        else:
            raise NotImplementedError()

    def fit(self, dataset: Dataset) -> None:
        x = dataset[self.x_key][:]
        if self.method == "variance":
            self._variance_filter(x, **self.kws)
        else:
            raise NotImplementedError()

    def _write_idx(self) -> None:
        if self.feature_idx is None:
            raise ValueError("No features have been fit yet")
        if self.feature_file is not None:
            self.feature_file.write_text("\n".join([str(i) for i in self.feature_idx]))

    def _read_idx(self) -> None:
        if self.feature_file is not None:
            self.feature_idx = [
                int(f) for f in self.feature_file.read_text().splitlines()
            ]

    def _variance_filter(self, x, quantile_threshold: float = 0.30) -> None:
        x = np.array(x)
        variance: np.ndarray = x.var(axis=0)
        if len(variance) == 0:
            logger.debug(variance)
            logger.debug(x)
            raise ValueError("variance is 0")
        thresh = np.quantile(variance, quantile_threshold)
        logger.info(f"threshold value: {thresh}")
        mask = variance >= thresh
        logger.info(f"{mask.sum()} features kept")
        idx = np.where(mask)[0]
        self.feature_idx = idx
        self._write_idx()

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)


def features_from_df(
    df: pl.DataFrame,
    convert_string_view: Literal["string", "large_string"] | None = "large_string",
) -> Features:
    from pyarrow import schema

    converted = Features.from_arrow_schema(schema(df.schema))
    if convert_string_view is not None:
        for k, v in converted.items():
            if v == Value("string_view"):
                converted[k] = Value(convert_string_view)
    return converted


def translate_df(
    df: pl.DataFrame,
    seq_col: str,
    new_col: str | None = None,
    degenerate_handling: Literal["ignore", "random", "error"] = "random",
) -> pl.DataFrame:
    new_col = new_col or f"{seq_col}_aa"

    def translate(seq) -> dict:
        dna: DNA = DNA(seq)
        degenerate = False
        if dna.has_degenerates():
            degenerate = True
            if degenerate_handling == "error":
                raise ValueError(
                    f"cannot translate degenerate bases in sequence `{seq}`"
                )
            elif degenerate_handling == "ignore":
                translated = None
            translated = next(dna.expand_degenerates()).translate()
        else:
            translated = dna.translate()
        return {new_col: str(translated), "dna_degenerate": degenerate}

    df = df.with_columns(pl.Series(map(translate, df[seq_col])).struct.unnest())
    return df


def add_random_cols(
    df: pl.DataFrame,
    cols: Sequence | None = None,
    choices: Sequence | None = None,
    low=None,
    high=None,
) -> pl.DataFrame:
    cols = cols or df.columns
    exprs = []
    for col in cols:
        if choices is None:
            ser = pl.Series(np.random.uniform(low, high, df.height)).alias(col)
        else:
            ser = pl.Series(np.random.choice(choices, df.height, replace=True)).alias(
                col
            )
        exprs.append(ser)
    return df.with_columns(*exprs)


def smoothen_log2(x: np.ndarray, l2_lower: int = -8, l2_upper: int = 10) -> np.ndarray:
    """Round values in x to their nearest values along the scale
    2**[l2_lower, l2_upper]
    """
    fit_to = np.arange(l2_lower, l2_upper, dtype=float)
    logged = np.log2(x)
    fill = np.zeros_like(x)
    rounded = logged.round()
    no_smoothing = logged == rounded
    fill[no_smoothing] = x[no_smoothing]
    for idx in np.argwhere(logged != rounded).flatten():
        to_smooth = logged[idx]
        nearest = np.argmin(np.abs(fit_to - to_smooth))
        fill[idx] = 2 ** fit_to[nearest]
    return fill
