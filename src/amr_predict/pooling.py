#!/usr/bin/env ipython
import contextlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias, override

import lightning as L
import polars as pl
import torch
import torch.nn as nn
from amr_predict.evaluation import MODEL_CLASSES
from amr_predict.models import MultiModule
from amr_predict.utils import ModuleConfig, load_as, read_tabular
from datasets import Array2D, Features, Value, concatenate_datasets
from datasets.arrow_dataset import Dataset
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torchmetrics.functional.pairwise import (
    pairwise_cosine_similarity,
    pairwise_euclidean_distance,
    pairwise_linear_similarity,
)

STATIC_POOLING_METHODS: TypeAlias = Literal[
    "sum", "mean", "similarity", "swe", "concat"
]
LEARNING_POOLING_METHODS: TypeAlias = Literal["autoencoder", "swe"]


class SeqPooler:
    def __init__(
        self,
        obs_keep: Sequence | None = None,
        embedding_key: str = "embedding",
        sample_key: str = "sample",
        sample_metadata: Path | str | None = None,
        sample_metadata_key: str | None = None,
        key: str = "x",
        **kws,
    ) -> None:
        """

        Parameters
        ----------
        obs_keep : Sequence
            Sequence of sample-level observations in the dataset to retain e.g. variables
            to predict. Sample names are automatically kept
        pooled_key : str
            Name of key (column) in dataset to keep the aggregated embeddings under
        """
        self.encoder: LabelEncoder = LabelEncoder()
        self.embedding_key: str = embedding_key
        self.obs_keep: list = list(obs_keep)
        self.sample_key: str = sample_key
        self.sample_metadata_key: str = sample_metadata_key or sample_key
        self.sample_metadata: Path | None = (
            Path(sample_metadata)
            if sample_metadata and isinstance(sample_metadata, str)
            else sample_metadata
        )
        self.key: str = key
        self.kws: dict = kws

    def _finalize_dataset(
        self,
        dataset: Dataset,
        aggregated: Tensor,
        mpath: Path | None = None,
        obs_keep: list | None = None,
    ) -> Dataset:
        """Aggregate variables from dataset by sample, and combine with pooled
        embeddings
        """
        mpath = mpath or self.sample_metadata
        meta: pl.DataFrame | None = read_tabular(mpath) if mpath else None
        obs_keep = obs_keep or self.obs_keep
        obs = dataset.to_polars()
        if meta is not None:
            obs = obs.join(
                meta.select([self.sample_metadata_key] + obs_keep),
                left_on=self.sample_key,
                right_on=self.sample_metadata_key,
                how="left",
            )
        obs = (
            obs.drop(self.embedding_key)
            .pipe(self._add_encoded)
            .group_by("encoded")
            .agg(pl.col(self.obs_keep + [self.sample_key]).first())
            .sort("encoded", descending=False)
            .drop("encoded")
        )
        to_concat = [
            Dataset.from_dict({self.key: aggregated}),
            Dataset.from_polars(obs),
        ]
        return concatenate_datasets(to_concat, axis=1).with_format("torch")

    def __call__(self, dataset: Dataset | Path | str) -> Dataset:
        d: Dataset = load_as(dataset) if not isinstance(dataset, Dataset) else dataset
        self.encoder.fit(d[self.sample_key])
        # Every method returns a tensor of the embeddings aggregated to sample level,
        # sorted in ascending order of the encoded sample names
        if self.method == "sum":
            x: Tensor = self._sum(d, mean=False)
        elif self.method == "mean":
            x = self._sum(d, mean=True)
        elif self.method == "similarity":
            x = self._similarity_weighted(d, **self.kws)
        return self._finalize_dataset(d, x)

    def _weights_from_pairwise(
        self,
        x: Tensor,
        mask: Tensor,
        fn: Callable = pairwise_cosine_similarity,
        pool: Literal["mean", "max", "sum"] = "mean",
    ) -> Tensor:
        weights = torch.zeros(x.shape[0])
        if pool == "mean":
            vals = fn(x[mask, :]).mean(axis=1)
        elif pool == "max":
            vals = fn(x[mask, :]).max(axis=1)
        elif pool == "sum":
            vals = fn(x[mask, :]).sum(axis=1)
        weights[mask] = vals
        return weights

    def _similarity_weighted(
        self,
        dataset: Dataset,
        metric: Literal["cosine", "dot_product", "euclidean"] = "cosine",
        pool: str = "mean",
    ):
        if metric == "cosine":
            fn = pairwise_cosine_similarity
        elif metric == "euclidean":
            fn = pairwise_euclidean_distance  # TODO: should change this to similarity
        elif metric == "dot_product":
            fn = pairwise_linear_similarity

        return self._sum(
            dataset,
            mean=False,
            weight_fn=lambda x, y: self._weights_from_pairwise(x, y, fn, pool=pool),
        )

    def _sum(
        self, dataset: Dataset, mean: bool = True, weight_fn: Callable | None = None
    ) -> Tensor:
        """Pool contig embeddings by summation, with multiple variants

        Parameters
        ----------
        weight_fn : Callable
            Function taking the whole embedding tensor and a boolean mask for the current sample
            Return a vector (of length equal to the columns of dataset)
            specifying how to weigh each embedding when summing to sample-level
        mean : bool
            For each sample, sum up embeddings then divide by the number of embeddings

        """
        samples = self._transform_samples(dataset)
        embeddings = dataset[self.embedding_key][:]
        unique_samples = torch.unique(samples, sorted=True)
        if weight_fn is None:
            mask = torch.stack([samples == s for s in unique_samples]).to(
                torch.get_default_dtype()
            )
        else:
            mask = torch.stack(
                [weight_fn(embeddings, samples == s) for s in unique_samples]
            )
        summed = torch.matmul(mask, embeddings)
        if mean:
            summed = torch.mul(summed, mask.sum(axis=1).reshape(-1, 1))
        return summed
