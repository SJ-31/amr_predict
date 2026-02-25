#!/usr/bin/env ipython
from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import batched
from pathlib import Path
from shutil import copyfile
from string import ascii_uppercase
from typing import Any, Literal, TypeAlias, override

import anndata as ad
import duckdb
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import skbio as sb
import torch
import torch.utils.data as td
from datasets import DatasetDict, Features, Value, concatenate_datasets
from datasets.arrow_dataset import Dataset
from datasets.load import load_from_disk
from loguru import logger
from numpy.random import Generator
from skbio import DNA
from sklearn.preprocessing import LabelEncoder
from torch import Tensor

CACHE_OPTIONS: TypeAlias = Literal["train_loss", "val_acc", "val_loss", "train_acc"]
PP_METHODS: TypeAlias = Literal["variance"]

logger.disable("amr_predict")


# * Utility functions

TASK_TYPES: TypeAlias = Literal["classification", "regression", "reconstruction"]


def sample_pairs(
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


def encode_strs(
    data: Dataset | LinkedDataset | ad.AnnData, task_names: tuple
) -> tuple[Dataset | ad.AnnData, dict[str, LabelEncoder]]:
    encoders = {}
    for task in task_names:
        encoder = LabelEncoder()
        if isinstance(data, Dataset):
            task_vec = data[task][:]
            data = data.remove_columns(task).add_column(
                task, encoder.fit_transform(task_vec)
            )
        elif isinstance(data, LinkedDataset):
            data.meta = data.meta.with_columns(
                pl.Series(encoder.fit_transform(task_vec)).alias(task)
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
    max_length: int,
    start_col: str = "Start",
    end_col: str = "Stop",
    indicate_ends: bool = True,
    prefix: str = "chunk",
) -> pl.DataFrame:
    """Split features into chunks if they exceed `max_length`

    Parameters
    ----------
    max_length : int
        Maximum feature length, and the length of the resulting chunks if features
        exceed this
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
    df = df.with_columns(length=pl.col(end_col) - pl.col(start_col)).with_columns(
        n_chunks=pl.col("length") // max_length,
        remaining=pl.col("length") % max_length,
    )

    def map_helper(x, col: Literal["start", "stop"] = "start"):
        nc = x["n_chunks"]
        start, stop = x[start_col], x[end_col]
        remaining = x["remaining"]
        has_remaining = remaining != 0 or remaining != x["length"]
        if col == "start":
            if not nc:
                return [start]
            val = [start + max_length * i for i in range(nc)]
            if has_remaining:
                val.append(start + max_length * nc)
            return val
        elif col == "stop":
            if not nc:
                return [stop]
            val = [start + max_length * (i + 1) for i in range(nc)]
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
    return df.drop(to_remove)


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


# * Cache
class EmbeddingCache:
    """Cache for batched text embeddings

    Can be used like a dictionary, but it's more efficient to use the "retrieve" method
    to look up queries in batched fashion
    EX: cache = EmbeddingCache(".cache")
        cache["ATCGACTA"] = [3, 9, 1, ...]
    """

    __slots__ = [
        "_storage",
        "_save_interval",
        "_prefix",
        "_dir",
        "_seen",
        "_with_tokens",
        "_rng",
        "_token_prop",
        "_seed",
    ]

    def retrieve(
        self, keys: pl.Series, tokens: bool = False, as_array: bool = True
    ) -> pl.DataFrame:
        key_length = len(keys)
        col = "token" if tokens else "seq"
        key_df = pl.DataFrame({"key": keys})
        lf: pl.LazyFrame = duckdb.query(f"""
        SELECT DISTINCT ON (t.key) t.key, t.{col}
        FROM '{self._glob()}' t
        INNER JOIN key_df k on k.key = t.key
        """).pl(lazy=True)
        if not lf.head(1).collect().height == 1:
            raise ValueError("None of the keys are present in the cache")
        collected: pl.DataFrame = (
            lf.collect() if not as_array else self._make_array(lf).collect()
        )
        if collected.shape[0] != key_length:
            logger.warning(f"""
            Number of keys {key_length} != shape of array {collected}
            Attempting to recover with lookups...
            """)
            in_collected = keys.is_in(collected["key"])
            if not in_collected.all():
                not_in = keys.filter(~in_collected).to_list()
                raise ValueError(
                    f"{len(not_in)} of the requested keys are missing from the cache\n{not_in}"
                )
            lookup: dict = dict(zip(collected["key"], collected[col]))
            return pl.DataFrame({"key": keys, col: [lookup[k] for k in keys]})
        return collected

    def _mask_in_df(
        self, df: pl.LazyFrame, column: str, mask_prop: float, height: int
    ) -> pl.DataFrame:
        return df.with_columns(
            pl.when(
                pl.Series(
                    self._rng.choice(
                        [True, False], p=[mask_prop, 1 - mask_prop], size=height
                    )
                )
            )
            .then(pl.lit(None))
            .otherwise(pl.col(column))
            .alias(column)
        )

    def __init__(
        self,
        dir: Path,
        prefix: str = "batch",
        save_interval: int = 10,
        with_tokens: bool = True,
        token_prop: float | None = None,
        seed: int | None = None,
    ) -> None:
        """

        Parameters
        ----------
        save_interval : int
            Number of times batches accumulate before writing a new parquet batch.
            duckdb recommends parquet files to be 100mb-10gb in size
        token_prop : float | None
            The proportion of tokens to randomly sample from each batch
            Likely necessary due to file size constraints
        seed : int | None
            Seed for random sampling operations
        """
        self._dir = dir
        self._token_prop: float | None = token_prop
        self._rng: Generator = np.random.default_rng(seed)
        self._seed: int = seed
        self._prefix = prefix
        self._save_interval: int = save_interval
        self._with_tokens = with_tokens
        try:
            _ = next(self._dir.glob(self._glob(False)))
            self._set_seen()
        except StopIteration:
            self._seen = set()
        self._storage: Tensor = torch.tensor([])

    def _set_seen(self) -> None:
        self._seen = set(
            duckdb.query(f"""
        SELECT key
        FROM '{self._glob()}'
        """)
            .pl()["key"]
            .to_list()
        )

    def rewrite(self, n_rows: int = 100_000, token_prop: float | None = None) -> None:
        "Read all entries into memory, remove duplicates and re-write cache to contain N parquet files"
        lf: pl.LazyFrame = self.to_pl().unique("seq")
        if token_prop:
            col = lf.select("token").collect()["token"].is_not_null()
            if col.any():
                lf = self._mask_in_df(lf, "token", 1 - token_prop, height=len(col))
        lf.sink_parquet(
            pl.PartitionMaxSize(
                base_path=self._dir,
                file_path=lambda x: x.full_path.parent.joinpath(
                    f"{self._prefix}_{x.file_idx}.parquet"
                ),
                max_size=n_rows,
            )
        )

    def _glob(self, with_dir: bool = True) -> str:
        if with_dir:
            return f"{str(self._dir)}/{self._prefix}*.parquet"
        return f"{self._prefix}*.parquet"

    def _make_array(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        seq_size, token_size = self._peek_size(lf)
        schema = lf.collect_schema()
        new_schema = {}
        if "seq" in schema:
            seq_type = schema["seq"].inner
            new_schema["seq"] = pl.Array(seq_type, seq_size)
        if "token" in schema:
            if lf.head(1).collect()["token"].item() is not None:
                token_type = schema["token"].inner.inner
                new_schema["token"] = pl.List(pl.Array(token_type, token_size))
        if not new_schema:
            raise ValueError("Neither `seq` nor `token` was present")
        return lf.cast(new_schema)

    def to_pl(self, as_array: bool = False) -> pl.LazyFrame:
        try:
            lf = (
                duckdb.query(f"""
        SELECT *
        FROM '{self._glob()}'
        """)
                .pl(lazy=True)
                .unique("key")
            )
        except duckdb.IOException:
            return pl.LazyFrame()
        if not as_array:
            return lf
        return self._make_array(lf)

    def __contains__(self, key: str) -> bool:
        return key in self._seen

    def __getitem__(self, key: str) -> Tensor:
        try:
            query = duckdb.query(f"""
            SELECT seq
            FROM '{self._glob()}'
            WHERE key = '{key}'
            """).fetchone()
            return torch.tensor(query[0])
        except duckdb.IOException:
            raise ValueError("no cached files available")

    def _peek_size(self, df: pl.LazyFrame) -> tuple[int, int]:
        cols = df.collect_schema().names()
        vals = df.head(1).collect()
        tk = vals["token"].item() if "token" in cols else None
        sq = len(vals["seq"].item()) if "seq" in cols else None
        return sq or 0, len(tk[0]) if tk is not None else 0

    def keys(self):
        return self._seen

    def __len__(self) -> int:
        return len(self._seen)

    @staticmethod
    def combine(
        caches: Sequence[EmbeddingCache | Path],
        new_path: Path,
        rewrite: bool = False,
        rewrite_kws: dict | None = None,
        **kws,
    ) -> EmbeddingCache:
        new: EmbeddingCache = EmbeddingCache(dir=new_path, **kws)
        new._dir.mkdir(exist_ok=True, parents=True)
        acc: int = 0
        for cache in caches:
            cache = EmbeddingCache(cache) if isinstance(cache, Path) else cache
            dir: Path = cache._dir
            for i, file in enumerate(dir.glob(cache._glob(False))):
                cur_index = i + acc
                new_loc = new._dir / f"{new._prefix}_{cur_index}.parquet"
                copyfile(file, new_loc)
                acc += i
        if rewrite:
            new.rewrite(**(rewrite_kws or {}))
        return new

    def save(
        self,
        to_embed: Sequence,
        fn: Callable[[Any], dict[str, Tensor]],
        batch_size: int = 50,
    ) -> None:
        """Embed all unique sequences in `to_embed`

        The embeddings can later be accessed by indexing the class instance like a dictionary

        Parameters
        ----------
        fn : Callable
            A function that takes a sequence of text and returns a
            dictionary mapping texts to a tuple of
                (sequence-level embeddings, token-level embeddings)
            The token-level embeddings (a 2D tensor) are optional
        """
        as_set = set(to_embed)
        n_old = len(as_set & self._seen)
        if n_old:
            logger.info(f"{n_old} found in cache")
        to_embed = as_set - self._seen
        logger.info(f"Embedding {len(to_embed)} new strings")
        counter = 0
        lfs = []
        dtype = torch2pl(torch.get_default_dtype())
        for batch in itertools.batched(set(to_embed), n=batch_size):
            try:
                schema: dict = {"key": pl.String}
                gen = fn(batch)
                key, seq, token = next(gen)
                if len(seq.shape) != 1:
                    raise ValueError("Embedding vectors must be 1D")
                if token is not None and len(token.shape) != 2:
                    raise ValueError("Token-level embeddings must all be 2D")

                schema["seq"] = pl.Array(dtype, len(seq))
                schema["token"] = (
                    pl.List(pl.Array(dtype, token.shape[1]))
                    if self._with_tokens
                    else pl.Null()
                )
                tmp = {"key": [], "seq": [], "token": []}
                gen = itertools.chain([(key, seq, token)], gen)
                # REVIEW: you didn't wanna have to do this, but had trouble with
                # casting types from the generator directly
                for k, s, t in gen:
                    tmp["key"].append(k)
                    tmp["seq"].append(s)
                    tmp["token"].append(t)

                lf = pl.LazyFrame(tmp, schema=schema)
                if not self._with_tokens:
                    lf = lf.with_columns(pl.lit(None).alias("token"))
                elif self._token_prop is not None:
                    lf = self._mask_in_df(lf, "token", 1 - self._token_prop, len(batch))
                self._seen |= set(lf.select("key").collect()["key"])
                lfs.append(lf)
                if counter == self._save_interval:
                    logger.info("Writing batch into cache")
                    self._write(lfs)
                    lfs = []
                    counter = 0
                else:
                    counter += 1
            except Exception as e:
                self._write(lfs)
                raise e

        self._write(lfs)

    def _write(self, lfs: list[pl.LazyFrame]) -> None:
        if lfs:
            file_count = len(list(self._dir.glob(self._glob(False))))
            save_path = self._dir.joinpath(f"{self._prefix}_{file_count}.parquet")
            pl.concat(lfs).sink_parquet(save_path)

    def to_dataset(
        self,
        df: pl.DataFrame,
        key_col: str,
        tokens: bool = False,
        new_col: str = "embedding",
        drop_null_columns: bool = True,
        hf: bool = False,
    ) -> Dataset | td.Dataset:
        if drop_null_columns:
            not_nulls = [s.name for s in df if not (s.null_count() == df.height)]
            nulls = [s for s in df.columns if s not in not_nulls]
            df = df[not_nulls]
            logger.warning(f"Dropping {len(nulls)} null columns: {nulls}")
        if hf:
            col = "token" if tokens else "seq"
            join_with = self.retrieve(df[key_col], tokens=tokens).rename({col: new_col})
            joined = df.join(join_with, left_on=key_col, right_on="key").filter(
                pl.col(new_col).is_not_null()
            )
            dset = Dataset.from_polars(joined).with_format("torch")
            # WARNING: the line above consumes a LOT of memory. But why? This is supposed to
            # be zero-copy
            return dset
        return LinkedDataset(
            meta=df, text_key=key_col, cache=self, token_level=tokens, x_key=new_col
        )


# ** Dataset integration


class LinkedDataset(td.Dataset):
    """Dataset using duckdb (through EmbeddingCache) to stream large embedding data in
    batches.

    Provides access to the embeddings for a sequence or token dataset
    Indexing returns dictionaries, like huggingface Datasets

    Parameters
    ----------
    meta : pl.DataFrame | Dataset
        Dataframe or dataset containing the metadata for each sequence
    text_key : str
        Key in the dictionary on indexing to store the embeddings

    Notes
    -----

    """

    def __init__(
        self,
        meta: pl.DataFrame | Dataset,
        cache: EmbeddingCache,
        token_level: bool = False,
        x_key: str = "embedding",
        text_key: str = "sequence",
    ) -> None:
        self.cache: EmbeddingCache = cache
        self.meta: pl.DataFrame = (
            meta if isinstance(meta, pl.DataFrame) else meta.to_polars()
        )
        self.text_key: str = text_key
        self.token_level: bool = token_level
        self.x_key: str = x_key
        super().__init__()

    @property
    def shape(self):
        n_col = self.meta.shape[1] + 1
        if not self.token_level:
            return self.meta.shape[0], n_col
        key_df = pl.DataFrame({"key": self.meta[self.text_key]})
        n_row = (
            duckdb.query(f"""
        SELECT DISTINCT ON (t.key) t.key, LEN(t.token)
        FROM '{self.cache._glob()}' t
        INNER JOIN key_df k on k.key = t.key
        """)
            .pl()["len(t.token)"]
            .sum()
        )
        return n_row, n_col

    def __len__(self):
        return self.meta.height

    def remove_missing(self):
        "Remove keys that are missing from the cache"
        logger.info("Size before removing missing keys: {}", self.meta.height)
        self.meta = self.meta.filter(pl.col(self.text_key).is_in(self.cache.keys()))
        logger.info("Size after: ", self.meta.height)

    def to_polars(self) -> pl.DataFrame:
        embeddings: pl.DataFrame = self.cache.retrieve(
            self.meta[self.text_key].unique(), tokens=self.token_level, as_array=True
        )
        joined = self.meta.join(
            embeddings, left_on=self.text_key, right_on="key", how="left"
        )
        if self.token_level:
            joined = joined.explode("token")
        return joined

    @property
    def columns(self):
        return self.meta.columns + [self.x_key]

    def _get_x(self, indices: Any | None = None) -> pl.DataFrame:
        df: pl.DataFrame = self.meta[indices] if indices is not None else self.meta
        x_df: pl.DataFrame = self.cache.retrieve(
            df[self.text_key].unique(), tokens=self.token_level, as_array=True
        )
        joined = df.join(x_df, left_on=self.text_key, right_on="key", how="left")
        return joined

    def _get_col(self, col) -> Tensor | pl.Series:
        if col == self.x_key:
            collected = self._get_x(None)
            if self.token_level:
                return collected["token"].to_torch()
            return collected["seq"].to_torch()
        return self.meta[col]

    @override
    def __getitem__(self, index) -> dict | Tensor | pl.Series:
        if isinstance(index, str):
            return self._get_col(index)
        level = "token" if self.token_level else "seq"
        df = self._get_x(index)
        if self.token_level:
            df = df.explode("token")
        can_convert = self.meta.select(pl.selectors.numeric()).columns
        converted = {col: df[col].to_torch() for col in can_convert}
        x = df[level].to_torch()
        result = df.drop([level, self.text_key] + can_convert).to_dict(as_series=False)
        result[self.x_key] = x
        result.update(converted)
        return result

    def __getitems__(self, indices) -> list:
        # Code taken from huggingface dataset, required to prevent nesting when using
        # torch DataLoader
        batch = self.__getitem__(indices)
        n_examples = len(batch[next(iter(batch))])
        return [
            {col: array[i] for col, array in batch.items()} for i in range(n_examples)
        ]

    def select(self, indices) -> LinkedDataset:
        return LinkedDataset(
            meta=self.meta[indices],
            cache=self.cache,
            token_level=self.token_level,
            x_key=self.x_key,
            text_key=self.text_key,
        )

    def _modify_columns(self, columns: Sequence, keep: bool) -> LinkedDataset:
        if isinstance(columns, str):
            columns = [columns]
        elif not isinstance(columns, list):
            columns = list(columns)
        if keep and self.text_key not in columns:
            columns.append(self.text_key)
        columns = [c for c in columns if c != self.x_key]
        kws = {
            "cache": self.cache,
            "x_key": self.x_key,
            "text_key": self.text_key,
            "token_level": self.token_level,
        }
        if keep:
            kws["meta"] = self.meta.select(columns)
        else:
            kws["meta"] = self.meta.drop(columns)
        return LinkedDataset(**kws)

    def select_columns(self, columns: Sequence) -> LinkedDataset:
        return self._modify_columns(columns, True)

    def remove_columns(self, columns: Sequence) -> LinkedDataset:
        return self._modify_columns(columns, False)

    def filter(self, fn: Callable) -> LinkedDataset:
        return LinkedDataset(
            meta=self.meta.filter(fn(self.meta)),
            cache=self.cache,
            x_key=self.x_key,
            text_key=self.text_key,
            token_level=self.token_level,
        )


# ** Cache utilities


def gen_from_cached(
    df: pl.DataFrame,
    key: str,
    cache: EmbeddingCache,
    keep: bool = False,
    drop_null_columns: bool = False,
    new_col: str = "embedding",
):
    """Return a generator function to produce a huggingface dataset

    Parameters
    ----------
    df : DataFrame
        Polars dataframe containing the query values that were embedded, as well as other
        metadata
    key : str
        Column of `df` containing the query values

    If `keep`, then the column containing the input to the embedding will be kept in
    the dataset
    """
    if drop_null_columns:
        df = df[[s.name for s in df if not (s.null_count() == df.height)]]

    def f():
        for row in df.iter_rows(named=True):
            if keep:
                embedding = cache[row[key]]
            else:
                embedding = cache[row.pop(key)]
            give = {new_col: embedding}
            give.update(row)
            yield give

    return f


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


def torch2pl(
    dtype: torch.dtype | Sequence[torch.dtype],
) -> pl.DataType | list[pl.DataType]:
    # Valid v2.9.1
    mapping: dict = {
        torch.bool: pl.Boolean,
        torch.int8: pl.Int8,
        torch.uint8: pl.UInt8,
        torch.int16: pl.Int16,  # alias: torch.short
        torch.uint16: pl.UInt16,
        torch.int32: pl.Int32,  # alias: torch.int
        torch.uint32: pl.UInt32,
        torch.int64: pl.Int64,  # alias: torch.long
        torch.uint64: pl.UInt64,
        torch.float32: pl.Float32,  # alias: torch.float
        torch.float64: pl.Float64,  # alias: torch.double
    }
    if isinstance(dtype, Sequence):
        converted = []
        for tp in dtype:
            try:
                converted.append(mapping[tp])
            except KeyError:
                raise ValueError(f"`{tp}` is not supported by polars")
        return converted
    try:
        return mapping[dtype]
    except KeyError:
        raise ValueError(f"`{dtype}` is not supported by polars")


def torch2hf(dtype: torch.dtype | Sequence[torch.dtype]) -> Value | list[Value]:
    # Valid v2.9.1
    mapping: dict = {
        torch.bool: "bool",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.int16: "int16",  # alias: torch.short
        torch.uint16: "uint16",
        torch.int32: "int32",  # alias: torch.int
        torch.uint32: "uint32",
        torch.int64: "int64",  # alias: torch.long
        torch.uint64: "uint64",
        torch.float16: "float16",  # alias: torch.half
        torch.float32: "float32",  # alias: torch.float
        torch.float64: "float64",  # alias: torch.double
    }
    if isinstance(dtype, Sequence):
        converted = []
        for tp in dtype:
            try:
                converted.append(Value(mapping[tp]))
            except KeyError:
                raise ValueError(f"`{tp}` is not supported by HF")
        return converted
    try:
        return Value(mapping[dtype])
    except KeyError:
        raise ValueError(f"`{dtype}` is not supported by HF")


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


DSET_TYPES: TypeAlias = Dataset | LinkedDataset | ad.AnnData | pl.DataFrame


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


# * Metadata functions


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


def get_ast_meta(cfg: dict) -> pl.DataFrame:
    args = cfg["ast_metadata"]
    df = read_tabular(args["file"])
    if not args.get("binarize"):
        df = df.with_columns(
            pl.any_horizontal(cs.ends_with("_class") == "resistant").alias(
                "any_resistant"
            ),
            pl.any_horizontal(cs.ends_with("_class") == "susceptible").alias(
                "any_susceptible"
            ),
        )
    else:
        df = df.with_columns(
            cs.ends_with("_class")
            .replace_strict({"resistant": 1}, default=0)
            .cast(pl.UInt32)
        ).with_columns(
            pl.any_horizontal(cs.ends_with("_class") == 1).alias("any_resistant")
        )
    if args.get("smooth"):
        for col in df.select(cs.by_dtype(pl.Float64)).columns:
            smoothed = smoothen_log2(df[col].to_numpy())
            df = df.with_columns(pl.Series(smoothed).alias(col))
    return df


def modify_metadata_test(
    meta_type: Literal["ast", "sample", "sequence"],
    original_df: pl.DataFrame,
    key_col: str,
    merging: pl.DataFrame,
) -> pl.DataFrame:
    if meta_type == "ast":
        cols = [
            col
            for col, dtype in original_df.schema.items()
            if not isinstance(dtype, pl.Boolean)
        ]
        class_cols = list(filter(lambda x: x.endswith("_class"), cols))
        other_cols = list(
            filter(lambda x: x != key_col and not x.endswith("_class"), cols)
        )
        if original_df.schema[class_cols[0]] == pl.String:
            merging = add_random_cols(
                merging, class_cols, ["resistant", "susceptible", "intermediate"]
            )
        else:
            merging = add_random_cols(merging, class_cols, [0, 1])
        merging = add_random_cols(merging, other_cols, low=0.01, high=1024)
    elif meta_type == "sequence":
        merging = add_random_cols(
            merging,
            cols=filter(lambda x: "gene" in x and x != "in_gene", original_df.columns),
            choices=list(ascii_uppercase)[:10],
        )
    else:
        merging = add_random_cols(
            merging,
            filter(lambda x: x != key_col, original_df.columns),
            choices=list(ascii_uppercase)[:15],
        )
    return merging


def with_metadata(
    dset: DSET_TYPES,
    cfg: dict,
    sample_col: str = "sample",
    meta_options: str | tuple[str, ...] = ("ast", "sample", "sequence", "external"),
    align: bool = False,
    dset_name: str | None = None,
) -> DSET_TYPES | tuple[DSET_TYPES, pl.DataFrame]:
    for_test: bool = cfg["test"]
    if isinstance(dset, ad.AnnData):
        merging: pl.DataFrame = pl.from_pandas(dset.obs)
    elif isinstance(dset, pl.DataFrame):
        merging = dset
    else:
        to_df = {sample_col: dset[sample_col][:]}
        if "sequence" in meta_options:
            to_df["uid"] = dset["uid"][:]
        merging = pl.DataFrame(to_df)
    if isinstance(meta_options, str):
        meta_options = [meta_options]
    for m in meta_options:
        if m == "sequence" and not dset_name:
            raise ValueError(
                "dataset name must be provided if requesting sequence metadata"
            )
        elif m == "sequence":
            start = cfg["out"]["tests"] if for_test else cfg["remote"]
            path = f"{start}/{cfg['in_date']}/datasets/processed_sequences/{dset_name}"
            df = load_as(path, "polars").with_columns(
                pl.any_horizontal(cs.contains("gene").is_not_null()).alias("in_gene")
            )
            key_col = "uid"
        elif m in {"ast", "sample"}:
            df = (
                get_ast_meta(cfg)
                if m == "ast"
                else read_tabular(cfg["sample_metadata"]["file"])
            )
            key_col = cfg[f"{m}_metadata"]["id_col"]
        elif m == "external":
            df = with_external_amr_predictions(
                merging, cfg, sample_col=sample_col, min_args=1
            )
            key_col = sample_col
        else:
            raise ValueError(
                f"metadata type `{m}` must be one of 'ast', 'sequence', 'sample', `external`"
            )
        merging = merging.join(
            df, left_on=sample_col, right_on=key_col, how="left", maintain_order="left"
        )
        if for_test:
            merging = modify_metadata_test(m, df, key_col, merging)
        tmp = (merging.null_count() / merging.height).unpivot()
        null_dict = dict(zip(tmp["variable"], tmp["value"]))
        # logger.info(
        #     "Percentage of nulls in merged metadata\n{}",
        #     null_dict,
        # )
    if not align and isinstance(dset, ad.AnnData):
        new = dset.copy()
        new.obs = merging.to_pandas()
        return new
    elif not align and isinstance(dset, pl.DataFrame):
        return merging
    elif not align:
        to_merge = Dataset.from_polars(merging.drop(sample_col))
        return concatenate_datasets([dset, to_merge], axis=1)
    return dset, merging


def read_format_hamr(
    cfg: dict, known_drugs: Sequence, sample_col: str = "sample"
) -> pl.DataFrame:
    hamr = pl.read_csv(
        cfg["seq_metadata"]["hamronization"],
        separator="\t",
        raise_if_empty=False,
        infer_schema_length=None,
    ).rename({"input_file_name": sample_col})
    to_remove = (
        "\\.mapping.*\\.deeparg",
        "\\.tsv\\.amrfinderplus",
        "\\.txt\\.rgi",
        "_retrieved-genes-.*",
    )
    wanted_cols = (sample_col, "analysis_software_name", "drug_class")
    drug_replacements = {"tobramcyin": "tobramycin", None: None}
    for pat in to_remove:
        hamr = hamr.with_columns(pl.col(sample_col).str.replace(pat, value=""))
    software = hamr["analysis_software_name"].unique()
    fmt = (
        hamr.select(wanted_cols)
        .with_columns(pl.col("drug_class").str.split(";"))
        .explode("drug_class")
        .with_columns(
            pl.col("drug_class")
            .str.to_lowercase()
            .str.strip_chars()
            .replace(drug_replacements)
        )
        .with_columns(
            pl.when(pl.col("drug_class").is_in(known_drugs))
            .then("drug_class")
            .when(pl.col("drug_class").is_null())
            .then(pl.lit(None))
            .otherwise(pl.lit("other_drug"))
        )
        .filter(pl.col("drug_class").is_not_null())
        .with_columns(pl.lit(1).alias("value"))
        .pivot(
            values="value",
            on=["analysis_software_name", "drug_class"],
            aggregate_function="sum",
        )
        .rename(
            lambda x: "_".join(
                x.removeprefix("{").removesuffix("}").replace('"', "").split(",")
            )
            if x.startswith("{")
            else x
        )
        .group_by(sample_col)
        .agg(pl.all().sum())
        .with_columns(
            *[
                pl.sum_horizontal(cs.starts_with(f"{sn}_")).alias(f"{sn}_any")
                for sn in software
            ]
        )
    )
    return fmt


def with_external_amr_predictions(
    df: pl.DataFrame, cfg: dict, sample_col: str = "sample", min_args: int = 1
) -> pl.DataFrame:
    """Add confusion matrix codes for predictions made by external AMR tools
    by cross-referencing with observed AST data

    Parameters
    ----------
    cfg : dict
        Dictionary as used in Snakemake pipeline
    min_args : int
        The miniumum number of resistance genes in the genome
        predicted by the tool to consider the tool as calling the genome to have an
        AMR phenotype

    Returns
    -------
    DataFrame where each sample is labeled as TP, TN, FP, FN for each tool in columns
    When possible, resistance predictions are checked with specific antibiotics e.g.
    `TOOL_tetracycline` is checked against the observed resistance to tetracycline.
    `TOOL_any` checks against the observed resistance to any antibiotic

    Notes
    -----
    The external tools are assumed to be aggregated by hAMRonization
    Label key:
    + False negative: no prediction but are resistant
    + False positive: predicted AMR genes but not resistant
    + True positive: predicted AMR genes and resistant
    + True negative: no prediction, not resistant

    TODO: what's a good value to set for min_args?
    TODO: you currently ignore the broad antibiotic classes in the hAMRonization tools,
        placing them all under `other`. Could instead try to classify the antibiotics
        in the AST data and then check against the tool prediction for that class
        e.g. `aminoglycoside antibiotic` would include streptomycin, gentamicin
        and amikacin
        See `broad_class_mapping` and finish it up
    WARNING: the cm codes assume that the AST data are ground truth
    """
    ast_data = get_ast_meta(cfg)
    broad_class_mapping = {"aminoglycoside": ("gentamicin", "amikacin", "streptomycin")}
    known_drugs = [
        col
        for col in ast_data.columns
        if not col.endswith("_class") and col not in {"BioSample", "Run"}
    ]
    # known_drugs.extend(broad_class_mapping.keys())
    df = df.select(sample_col).join(
        ast_data, how="left", left_on=sample_col, right_on=cfg["ast_metadata"]["id_col"]
    )
    hamr = read_format_hamr(cfg, known_drugs, sample_col=sample_col)
    merged = df.join(hamr, on=sample_col, how="left")
    res_val = 1 if cfg["ast_metadata"].get("binarize") else "resistant"
    for col in hamr.columns:
        drug = col.split("_")[1] if "_" in col else ""
        pass_thresh = pl.col(col) >= min_args

        def code_cm(df: pl.DataFrame, expr) -> pl.DataFrame:
            return df.with_columns(
                pl.when(pass_thresh & expr)
                .then(pl.lit("TP"))
                .when(pass_thresh)
                .then(pl.lit("FP"))
                .when(expr)
                .then(pl.lit("FN"))
                .otherwise(pl.lit("TN"))
                .alias(f"{col}_cm")
            )

        if col == sample_col:
            continue
        elif col.endswith("_any"):
            merged = code_cm(merged, pl.col("any_resistant"))
        elif drug_cols := broad_class_mapping.get(drug):
            any_expr = pl.any_horizontal(cs.by_name(drug_cols) == res_val)
            merged = code_cm(merged, any_expr)
        elif f"{drug}_class" in merged.columns:
            merged = code_cm(merged, pl.col(f"{drug}_class") == res_val)
    return merged
