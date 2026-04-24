#!/usr/bin/env ipython

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path
from shutil import copyfile
from typing import Any, Callable, Literal, get_args, override

import duckdb
import jaxtyping
import numpy as np
import polars as pl
import torch
import torch.utils.data as td
from amr_predict.compat import torch2pl
from amr_predict.pooling import BasicPoolings, pool_tensor
from attrs import define, field, validators
from beartype import beartype
from datasets import Dataset
from loguru import logger
from numpy.random import Generator
from torch import Tensor

LEVELS = Literal["logits", "seqs", "tokens"]


@define
class EmbeddingCache:
    """Cache for batched text embeddings

    Can be used like a dictionary, but it's more efficient to use the "retrieve" method
    to look up queries in batched fashion
    EX: cache = EmbeddingCache(".cache")
        cache["ATCGACTA"] = [3, 9, 1, ...]
    """

    dir: Path
    prefix: str = "batch"
    rng: Generator = field(
        default=4243, converter=lambda val: np.random.default_rng(val)
    )
    save_proba: bool = False
    token_amount: float | int | None = None
    save_mode: Literal["both", "seqs", "tokens"] = field(
        default="seqs", validator=validators.in_(["both", "seqs", "tokens"])
    )
    save_interval: int = 10
    seen: set = field(init=False, factory=set)
    pooling: BasicPoolings = BasicPoolings.MEAN
    storage: Tensor = field(init=False, factory=lambda: torch.tensor([]))

    def __attrs_post_init__(self):
        try:
            _ = next(self.dir.glob(self._glob(False)))
            self._set_seen()
        except StopIteration:
            pass

    @beartype
    def retrieve(
        self, keys: pl.Series, level: LEVELS, as_array: bool = True
    ) -> pl.DataFrame:
        key_length = len(keys)
        col = level.removesuffix("s")
        col_selector = f"t.{col}"
        if level == "tokens":
            col_selector = f"t.{col}, t.token_idx"
        key_df = pl.DataFrame({"key": keys})
        lf: pl.LazyFrame = duckdb.query(f"""
        SELECT DISTINCT ON (t.key) t.key, {col_selector}
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
                    self.rng.choice(
                        [True, False], p=[mask_prop, 1 - mask_prop], size=height
                    )
                )
            )
            .then(pl.lit(None))
            .otherwise(pl.col(column))
            .alias(column)
        )

    def _maybe_sample_tokens(
        self, tokens: jaxtyping.Float[Tensor, "a b"]
    ) -> tuple[Tensor, np.ndarray | list]:
        n_tokens = tokens.shape[0]
        if self.token_amount is None or (
            isinstance(self.token_amount, int) and self.token_amount >= n_tokens
        ):
            return tokens, list(range(n_tokens))
        elif isinstance(self.token_amount, float):
            n = max(1, int(n_tokens * self.token_amount))
        else:
            n = self.token_amount
        indices = self.rng.choice(range(n_tokens), n, replace=False)
        return tokens[indices, :], indices

    def _set_seen(self) -> None:
        self.seen = set(
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
                base_path=self.dir,
                file_path=lambda x: x.full_path.parent.joinpath(
                    f"{self.prefix}_{x.file_idx}.parquet"
                ),
                max_size=n_rows,
            )
        )

    def _glob(self, with_dir: bool = True) -> str:
        if with_dir:
            return f"{str(self.dir)}/{self.prefix}*.parquet"
        return f"{self.prefix}*.parquet"

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
        return key in self.seen

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
        return self.seen

    def __len__(self) -> int:
        return len(self.seen)

    @staticmethod
    def combine(
        caches: Sequence[EmbeddingCache | Path],
        new_path: Path,
        rewrite: bool = False,
        rewrite_kws: dict | None = None,
        **kws,
    ) -> EmbeddingCache:
        new: EmbeddingCache = EmbeddingCache(dir=new_path, **kws)
        new.dir.mkdir(exist_ok=True, parents=True)
        acc: int = 0
        for cache in caches:
            cache = EmbeddingCache(cache) if isinstance(cache, Path) else cache
            dir: Path = cache.dir
            for i, file in enumerate(dir.glob(cache._glob(False))):
                cur_index = i + acc
                new_loc = new.dir / f"{new.prefix}_{cur_index}.parquet"
                copyfile(file, new_loc)
                acc += i
        if rewrite:
            new.rewrite(**(rewrite_kws or {}))
        return new

    def save(
        self,
        to_embed: Sequence,
        embed_fn: Callable,
        batch_size: int = 50,
    ) -> None:
        """Embed all unique sequences in `to_embed`

        The embeddings can later be accessed by indexing the class instance like a dictionary

        Parameters
        ----------
        embedding_fn : Callable
            A function that takes a sequence of strs and returns a
            dictionary mapping strs to a tuple of
                (str, token-level embeddings, token probabilities)
            Token probabilities are only required if self.with_proba is set
        """
        as_set = set(to_embed)
        n_old = len(as_set & self.seen)
        if n_old:
            logger.info(f"{n_old} found in cache")
        to_embed = as_set - self.seen
        logger.info(f"Embedding {len(to_embed)} new strings")
        counter, lfs = 0, []
        dtype = torch2pl(torch.get_default_dtype())
        token: jaxtyping.Float[Tensor, "a b"]

        batches = itertools.batched(set(to_embed), n=batch_size)
        try:
            first_batch = next(batches)
        except StopIteration:
            return
        schema: dict = {"key": pl.String}
        save_into = {"key": []}
        _, token, logits = next(embed_fn(first_batch))
        assert isinstance(token, jaxtyping.Float[Tensor, "a b"])
        if self.save_proba:
            assert isinstance(logits, jaxtyping.Float[Tensor, "a"])

        if self.save_mode in ("seqs", "both"):
            schema["seq"] = pl.Array(dtype, token.shape[1])
            save_into["seq"] = []
        if self.save_mode in ("tokens", "both"):
            schema["token"] = pl.List(pl.Array(dtype, token.shape[1]))
            save_into["token"] = []
            schema["token_idx"] = pl.List(pl.Int64)
            save_into["token_idx"] = []
        if self.save_proba:
            schema["token_pr"] = pl.List(dtype)
            save_into["token_pr"] = []

        for batch in itertools.chain([first_batch], batches):
            tmp = save_into.copy()
            try:
                gen = embed_fn(batch)
                # REVIEW: you didn't wanna have to do this, but had trouble with
                # casting types from the generator directly
                for k, t, l in gen:
                    tmp["key"].append(k)
                    if self.save_mode in ("seqs", "both"):
                        tmp["seq"].append(pool_tensor(t, method=self.pooling))
                    if self.save_mode in ("tokens", "both"):
                        tokens, indices = self._maybe_sample_tokens(t)
                        tmp["token"].append(tokens)
                        tmp["token_idx"].append(indices)
                    if self.save_proba:
                        tmp["token_pr"].append(l)
                lf = pl.LazyFrame(tmp, schema=schema)
                self.seen |= set(tmp["key"])
                lfs.append(lf)
                if counter == self.save_interval:
                    logger.info("Writing batch into cache")
                    self._write(lfs)
                    lfs, counter = [], 0
                else:
                    counter += 1
            except Exception as e:
                self._write(lfs)
                raise e

        self._write(lfs)

    def _write(self, lfs: list[pl.LazyFrame]) -> None:
        if lfs:
            file_count = len(list(self.dir.glob(self._glob(False))))
            save_path = self.dir.joinpath(f"{self.prefix}_{file_count}.parquet")
            pl.concat(lfs).sink_parquet(save_path)

    def to_dataset(
        self,
        df: pl.DataFrame,
        key_col: str,
        level: LEVELS,
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
            col = level.removesuffix("s")
            join_with = self.retrieve(df[key_col], level=level).rename({col: new_col})
            joined = df.join(join_with, left_on=key_col, right_on="key").filter(
                pl.col(new_col).is_not_null()
            )
            dset = Dataset.from_polars(joined).with_format("torch")
            # WARNING: the line above consumes a LOT of memory. But why? This is supposed to
            # be zero-copy
            return dset
        return LinkedDataset(
            meta=df, text_key=key_col, cache=self, level=level, x_key=new_col
        )


@define
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
    """

    cache: EmbeddingCache
    meta: pl.DataFrame = field(
        converter=lambda x: x if isinstance(x, pl.DataFrame) else x.to_polars()
    )
    level: LEVELS = field(default="seqs", validator=validators.in_(get_args(LEVELS)))
    x_key: str = "embedding"
    text_key: str = "sequence"

    @property
    def shape(self):
        n_col = self.meta.shape[1] + 1
        if self.level != "tokens":
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

    def sample(self, by: str | None = None, **kws) -> None:
        """Reduce the number of keys in the dataset"""
        if by is None:
            self.meta = self.meta.sample(**kws)
        else:
            idx = (
                self.meta.with_row_index()
                .group_by(by)
                .agg(pl.col("index"))
                .with_columns(pl.col("index").list.sample(**kws))
                .explode("index")["index"]
            )
            self.meta = self.meta[idx, :]

    def __len__(self):
        return self.shape[0]

    def remove_missing(self):
        "Remove keys that are missing from the cache"
        logger.info("Size before removing missing keys: {}", self.meta.height)
        keys = set(self.cache.keys())
        self.meta = self.meta.filter(pl.col(self.text_key).is_in(keys))
        logger.info("Size after: ", self.meta.height)

    def to_polars(self) -> pl.DataFrame:
        embeddings: pl.DataFrame = self.cache.retrieve(
            self.meta[self.text_key].unique(), level=self.level, as_array=True
        )
        joined = self.meta.join(
            embeddings, left_on=self.text_key, right_on="key", how="left"
        )
        if self.level == "tokens":
            joined = joined.explode("token", "token_idx")
        return joined

    @property
    def columns(self):
        return self.meta.columns + [self.x_key]

    def _get_x(self, indices: Any | None = None) -> pl.DataFrame:
        df: pl.DataFrame = self.meta[indices] if indices is not None else self.meta
        x_df: pl.DataFrame = self.cache.retrieve(
            df[self.text_key].unique(), level=self.level, as_array=True
        )
        joined = df.join(
            x_df, left_on=self.text_key, right_on="key", how="left", validate="m:1"
        )
        return joined

    def _get_col(self, col) -> Tensor | pl.Series:
        if col == self.x_key:
            collected = self._get_x(None)
            return collected[self.level.removesuffix("s")].to_torch()
        return self.meta[col]

    @override
    def __getitem__(self, index) -> dict | Tensor | pl.Series:
        if isinstance(index, str):
            return self._get_col(index)
        level = self.level.removesuffix("s")
        df = self._get_x(index)
        if self.level == "tokens":
            df = df.explode("token", "token_idx")
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
            level=self.level,
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
            "level": self.level,
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
            level=self.level,
        )


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
