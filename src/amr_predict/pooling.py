#!/usr/bin/env ipython
import contextlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias, override

import lightning as L
import polars as pl
import torch
import torch.nn as nn
from amr_predict.models import BaseNN
from amr_predict.utils import LinkedDataset, ModuleConfig, load_as, read_tabular
from datasets import Array2D, Features, Value, concatenate_datasets
from datasets.arrow_dataset import Dataset
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torchmetrics.functional.pairwise import (
    pairwise_cosine_similarity,
    pairwise_euclidean_distance,
    pairwise_linear_similarity,
)

logger.disable("amr_predict")

STATIC_POOLING_METHODS: TypeAlias = Literal[
    "sum", "mean", "similarity", "swe", "concat", "seq_subset"
]
LEARNING_POOLING_METHODS: TypeAlias = Literal["autoencoder", "swe"]


class SeqPooler:
    def __init__(
        self,
        obs_keep: Sequence | None = None,
        embedding_key: str = "embedding",
        sample_key: str = "sample",
        whitelist_col: str | None = None,
        feature_whitelist: tuple = (),
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
        whitelist_col : str | None
            Name of column to use for feature whitelist
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
        self.whitelist_col: str | None = whitelist_col
        self.feature_whitelist: tuple = feature_whitelist
        self.key: str = key
        self.kws: dict = kws

    def apply_whitelist(
        self, dataset: Dataset | LinkedDataset
    ) -> Dataset | LinkedDataset:
        if isinstance(dataset, Dataset):
            return dataset.filter(
                lambda x: x[self.whitelist_col] in self.feature_whitelist
            )
        return dataset.filter(
            lambda x: x[self.whitelist_col].isin(self.feature_whitelist)
        )

    def _finalize_dataset(
        self,
        dataset: Dataset | LinkedDataset,
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

    def _encode_samples(self, dataset: Dataset) -> Tensor:
        return torch.tensor(self.encoder.transform(dataset[self.sample_key][:]))

    def _add_encoded(self, df: pl.DataFrame) -> pl.DataFrame:
        encoded = self.encoder.transform(df[self.sample_key])
        return df.with_columns(pl.Series(encoded).alias("encoded"))

    def _pad_embeddings(self, dataset: Dataset) -> Tensor:
        """
        Pads so that every genome has the same number of embedding sequences
        `by_sample` has shape n_samples x max_n_embeddings x embedding_size
        """
        embeddings: Tensor = dataset[self.embedding_key][:]
        samples = self._encode_samples(dataset)
        unique_samples = torch.unique(samples, sorted=True)
        by_sample = nn.utils.rnn.pad_sequence(
            [embeddings[samples == s, :] for s in unique_samples], batch_first=True
        )
        return by_sample


class StaticPooler(SeqPooler):
    """Class to pool samples' contig embeddings into a single genome-scale embedding"""

    def __init__(
        self,
        obs_keep: Sequence | None = None,
        embedding_key: str = "embedding",
        method: STATIC_POOLING_METHODS = "mean",
        sample_key: str = "sample",
        sample_metadata: Path | str | None = None,
        sample_metadata_key: str | None = None,
        whitelist_col: str | None = None,
        feature_whitelist: tuple = (),
        key: str = "x",
        **kws,
    ) -> None:
        super().__init__(
            obs_keep,
            embedding_key=embedding_key,
            sample_key=sample_key,
            sample_metadata=sample_metadata,
            sample_metadata_key=sample_metadata_key,
            key=key,
            whitelist_col=whitelist_col,
            feature_whitelist=feature_whitelist,
            **kws,
        )
        self.method: STATIC_POOLING_METHODS = method

    def __call__(self, dataset: Dataset | Path | str) -> Dataset:
        d: Dataset = load_as(dataset) if not isinstance(dataset, Dataset) else dataset
        if self.whitelist_col is not None and self.feature_whitelist:
            d = self.apply_whitelist(d)
        self.encoder.fit(d[self.sample_key][:])
        # Every method returns a tensor of the embeddings aggregated to sample level,
        # sorted in ascending order of the encoded sample names
        if self.method == "sum":
            x: Tensor = self._sum(d, weigh=False)
        elif self.method == "mean":
            x = self._sum(d, weigh=True)
        elif self.method == "similarity":
            x = self._similarity_weighted(d, **self.kws)
        elif self.method == "swe":
            x = self._swe(d, **self.kws)
        elif self.method == "seq_subset":
            x = self._seq_subset(d, **self.kws)
        elif self.method == "concat":
            padded = self._pad_embeddings(d)
            x = padded.reshape(padded.shape[0], -1)
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
            weigh=False,
            weight_fn=lambda x, y: self._weights_from_pairwise(x, y, fn, pool=pool),
        )

    def _swe(self, dataset: Dataset, m: int = 1000, n_out: int = -1) -> Tensor:
        """Pool contig embeddings with frozen swe

        Parameters
        ----------
        m : int
            the number of points in the reference set
        n_out : int
            output size of the pooled embeddings. By default, does not change the
            shape of the original sequence embeddings

        """
        _, d_in = dataset[self.embedding_key][:].shape
        num_slices = d_in if n_out == -1 else n_out
        pooler = SWE_Pooling(
            d_in=d_in, num_slices=num_slices, num_ref_points=m, freeze_swe=True
        )
        return pooler(self._pad_embeddings(dataset))

    def _seq_subset(
        self,
        dataset: LinkedDataset | Dataset,
        priority: list,
        subset_col: str,
        agg: Literal["mean", "max", None] = None,
        split: str | None = None,
        rng: int | None = None,
    ) -> Tensor:
        """
        Pooling method that selects a specific sequence to use as representative of
        the sample, or aggregates a subset of sequences

        Parameters
        ----------
        priority : list
            List of sample labels in subset_col, in order of decreasing importance,
            to use for selecting sequences of a sample
            EX: passing the list ["geneK", "mucin", "pump"], embeddings from
            sequences labeled as "geneK" in `subset_col` will be prioritized
            If for a given sample, none of its sequences are in `priority`, they are all
            aggregated or a random sequence is chosen
        agg : Aggregation method when multiple sequences bear the highest priority
            label
        rng : int
            Random seed for sampling
        """
        meta: pl.DataFrame = (
            dataset.meta if isinstance(dataset, LinkedDataset) else dataset.to_polars()
        )
        not_in_meta = set(priority) - set(meta[subset_col])
        if len(not_in_meta) > 0:
            logger.warning(
                f"The following entries of `priority` {not_in_meta} are not present in the dataset and will be removed. Removing..."
            )
            priority = [p for p in priority if p not in not_in_meta]
        if not priority:
            raise ValueError(
                f"No values specified in `priority` were found in column `{subset_col}`"
            )
        if split is not None:
            meta = meta.with_columns(pl.col(subset_col).str.split(split))
            list_check = True
        else:
            list_check = False

        mapping = {p: len(priority) - i for i, p in enumerate(priority)}

        def score_list(lst: list) -> int:
            return max(map(lambda x: mapping.get(x, 0), lst))

        meta = meta.with_row_index()
        samples = self._encode_samples(dataset)
        embeddings = dataset[self.embedding_key][:]
        unique_samples = torch.unique(samples, sorted=True)
        tmp = []
        # If the current sample is `null` for subset col, take a random element
        # Otherwise, take elements according to the priority list
        if not list_check:
            meta = meta.with_columns(
                pl.col(subset_col).replace_strict(mapping, default=0).alias("score")
            )
        else:
            meta = meta.with_columns(
                pl.col(subset_col)
                .map_elements(
                    lambda x: score_list(x) if x is not None else 0,
                    return_dtype=pl.Int32,
                )
                .alias("score")
            )

        for sample in unique_samples:
            mask = dataset[self.sample_key][:] == sample
            current = meta.filter(mask)
            current = current.filter(pl.col("score") == current["score"].max())
            if agg is None:
                sampled = current.sample(1, seed=rng)["index"]
                tmp.append(embeddings[sampled, :])
            elif agg == "mean":
                tmp.append(embeddings[mask].sum(dim=1))
            elif agg == "max":
                tmp.append(embeddings[mask].max(dim=1)[0])
        return torch.stack(tmp)

    def _sum(
        self, dataset: Dataset, weigh: bool = True, weight_fn: Callable | None = None
    ) -> Tensor:
        """Pool contig embeddings by summation, with multiple variants

        Parameters
        ----------
        weight_fn : Callable
            Function taking the whole embedding tensor and a boolean mask for the current sample
            Return a vector (of length equal to the columns of dataset)
            specifying how to weigh each embedding when summing to sample-level
        weigh : bool
            Multiply the pooled embeddings by some weight vector

        """
        samples = self._encode_samples(dataset)
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
        if weigh:
            summed = torch.mul(summed, mask.sum(axis=1).reshape(-1, 1))
        return summed


# * Parametrizable

# TODO: want two methods here: swe and autoencoder


class LearningPooler(SeqPooler, L.LightningModule):
    def __init__(
        self,
        out_features: int,
        obs_keep: Sequence | None = None,
        method: LEARNING_POOLING_METHODS = "swe",
        model: L.LightningModule | nn.Module | None = None,
        embedding_key: str = "embedding",
        sample_key: str = "sample",
        sample_metadata: Path | str | None = None,
        sample_metadata_key: str | None = None,
        whitelist_col: str | None = None,
        feature_whitelist: tuple = (),
        key: str = "x",
        **kws,
    ) -> None:
        super().__init__(
            obs_keep=obs_keep,
            embedding_key=embedding_key,
            sample_key=sample_key,
            sample_metadata=sample_metadata,
            sample_metadata_key=sample_metadata_key,
            key=key,
            whitelist_col=whitelist_col,
            feature_whitelist=feature_whitelist,
            **kws,
        )
        self.method: LEARNING_POOLING_METHODS = method
        if method == "swe":
            kws["d_in"] = out_features
            self.pooler: nn.Module | L.LightningModule = SWE_Pooling(**kws)
            if not model:
                raise ValueError(
                    "Prediction model must be given if using `swe` pooling"
                )
        elif method == "autoencoder":
            kws["out_features"] = out_features
            self.pooler = AePooling(**kws)
        self.embedding_size: int
        self.out_features: int = out_features
        self.model: BaseNN | None = model

    def format_for_training(self, dataset: Dataset, obs_types: dict) -> Dataset:
        """
        Modify the shapes of embeddings and observations to be compatible with the
        chosen pooling method
        Must be used before any training
        """
        self.encoder.fit(dataset[self.sample_key][:])
        padded = self._pad_embeddings(dataset)
        self.embedding_size = padded.shape[1]
        if self.sample_key not in obs_types:
            obs_types[self.sample_key] = "string"
        grouped = (
            dataset.select_columns(list(obs_types.keys()))
            .to_polars()
            .pipe(self._add_encoded)
            .group_by(self.sample_key)
            .agg(pl.all().first())
            .sort("encoded", descending=False)
            .drop("encoded")
            .to_dict()
        )
        features = Features(
            dict(
                x=Array2D(shape=(padded[0, :, :].shape), dtype="float32"),
                **{o: Value(v) for o, v in obs_types.items()},
            )
        )
        batched_ds = Dataset.from_dict(
            dict(x=padded, **grouped), features=features
        ).with_format("torch")
        return batched_ds

    def _pool(self, batch, with_tasks: bool = False) -> dict:
        embeddings: Tensor = batch[self.embedding_key][:]
        # Tensor of shape (batch, n_contigs, embedding_size)
        # After dataset preparation the number of contigs will be equalized across the
        # dataset by padding
        pooled: Tensor = self.pooler(embeddings)
        if with_tasks and self.model is not None:
            to_send = {self.model.x_key: pooled}
            to_send.update({t: batch[t] for t in self.model.task_names})
        return to_send

    @override
    def training_step(self, batch, batch_idx):
        if self.method == "swe":
            return self.model.training_step(self._pool(batch, True), batch_idx)
        elif self.method == "autoencoder":
            return self.model.training_step(batch, batch_idx)

    @override
    def predict_step(self, batch, batch_idx) -> Any:
        if self.method == "swe":
            return self.model.predict_step(self._pool(batch, False), batch_idx)


# * Models
# ** SWE
"""
@article{naderializadeh2025_plm_swe,
  title={Aggregating residue-level protein language model embeddings with optimal transport},
  author={NaderiAlizadeh, Navid and Singh, Rohit},
  journal={Bioinformatics Advances},
  volume={5},
  number={1},
  pages={vbaf060},
  year={2025},
  publisher={Oxford University Press}
}
"""


class Interp1d(torch.autograd.Function):
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {"x": x, "y": y, "xnew": xnew}.items():
            assert len(vec.shape) <= 2, "interp1d: all inputs must be " "at most 2-D."
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, "All parameters must be on the same device."
        device = device[0]

        # Checking for the dimensions
        assert v["x"].shape[1] == v["y"].shape[1] and (
            v["x"].shape[0] == v["y"].shape[0]
            or v["x"].shape[0] == 1
            or v["y"].shape[0] == 1
        ), (
            "x and y must have the same number of columns, and either "
            "the same number of row or one of them having only one "
            "row."
        )

        reshaped_xnew = False
        if (
            (v["x"].shape[0] == 1)
            and (v["y"].shape[0] == 1)
            and (v["xnew"].shape[0] > 1)
        ):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v["xnew"].shape
            v["xnew"] = v["xnew"].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v["x"].shape[0], v["xnew"].shape[0])
        shape_ynew = (D, v["xnew"].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0] * shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v["xnew"].shape[0] == 1:
            v["xnew"] = v["xnew"].expand(v["x"].shape[0], -1)

        torch.searchsorted(v["x"].contiguous(), v["xnew"].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v["x"].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ["x", "y", "xnew"]:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [
                    None,
                ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat["slopes"] = is_flat["x"]
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v["slopes"] = (v["y"][:, 1:] - v["y"][:, :-1]) / (
                eps + (v["x"][:, 1:] - v["x"][:, :-1])
            )

            # now build the linear interpolation
            ynew = sel("y") + sel("slopes") * (v["xnew"] - sel("x"))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
            ctx.saved_tensors[0],
            [i for i in inputs if i is not None],
            grad_out,
            retain_graph=True,
        )
        result = [
            None,
        ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)


class SWE_Pooling(nn.Module):
    def __init__(self, d_in, num_slices, num_ref_points, freeze_swe=False):
        """
        Produces fixed-dimensional permutation-invariant embeddings for input sets of arbitrary size based on sliced-Wasserstein embedding.
        Inputs:
            d_in:  The dimensionality of the space that each set sample belongs to
            num_ref_points: Number of points in the reference set
            num_slices: Number of slices
        """
        super(SWE_Pooling, self).__init__()
        self.d_in = d_in
        self.num_ref_points = num_ref_points
        self.num_slices = num_slices

        uniform_ref = (
            torch.linspace(-1, 1, num_ref_points).unsqueeze(1).repeat(1, num_slices)
        )
        self.reference = nn.Parameter(
            uniform_ref
        )  # initalize the references using a uniform distribution

        self.theta = nn.utils.weight_norm(
            nn.Linear(d_in, num_slices, bias=False), dim=0
        )

        self.theta.weight_g.data = torch.ones_like(
            self.theta.weight_g.data, requires_grad=False
        )
        self.theta.weight_g.requires_grad = False

        nn.init.normal_(
            self.theta.weight_v
        )  # initalize the slicers using a Gaussian distribution

        if freeze_swe:  # freezing the slicer and reference parameters
            self.theta.weight_v.requires_grad = False
            self.reference.requires_grad = False

        # weights to reduce the output embedding dimensionality
        self.weight = nn.Linear(num_ref_points, 1, bias=False)

    def forward(self, X, mask=None):
        """
        Calculates SWE pooling of X over its second dimension (i.e., sequence length)

        Input:
            X:  B x N x d_in tensor, containing a batch of B sequences, each containing N embeddings in a d_in-dimensional space
            mask [optional]: B x N binary tensor, with 1 iff the sequence element is valid; used for the case where sequence lengths are different
        Output:
            weighted_embeddings: B x num_slices tensor, containing a batch of B pooled embeddings, each of dimension "num_slices" (i.e., number of slices)
        """

        B, N, _ = X.shape
        Xslices = self.get_slice(X)

        M, _ = self.reference.shape

        if mask is None:
            # serial implementation should be used if set sizes are different and no input mask is provided
            Xslices_sorted, Xind = torch.sort(Xslices, dim=1)

            if M == N:
                Xslices_sorted_interpolated = Xslices_sorted
            else:
                x = (
                    torch.linspace(0, 1, N + 2)[1:-1]
                    .unsqueeze(0)
                    .repeat(B * self.num_slices, 1)
                    .to(X.device)
                )
                xnew = (
                    torch.linspace(0, 1, M + 2)[1:-1]
                    .unsqueeze(0)
                    .repeat(B * self.num_slices, 1)
                    .to(X.device)
                )
                y = torch.transpose(Xslices_sorted, 1, 2).reshape(
                    B * self.num_slices, -1
                )
                Xslices_sorted_interpolated = torch.transpose(
                    Interp1d()(x, y, xnew).view(B, self.num_slices, -1), 1, 2
                )
        else:
            # replace invalid set elements with points to the right of the maximum element for each slice and each set (which will not impact the sorting and interpolation process)
            invalid_elements_mask = ~mask.bool().unsqueeze(-1).repeat(
                1, 1, self.num_slices
            )
            Xslices_copy = Xslices.clone()
            Xslices_copy[invalid_elements_mask] = -1e10

            top2_Xslices, _ = torch.topk(Xslices_copy, k=2, dim=1)
            max_Xslices = top2_Xslices[:, 0].unsqueeze(1)
            delta_y = -torch.diff(top2_Xslices, dim=1)

            Xslices_modified = Xslices.clone()

            Xslices_modified[invalid_elements_mask] = max_Xslices.repeat(1, N, 1)[
                invalid_elements_mask
            ]

            delta_x = 1 / (1 + torch.sum(mask, dim=1, keepdim=True))
            slope = delta_y / delta_x.unsqueeze(-1).repeat(
                1, 1, self.num_slices
            )  # B x 1 x num_slices
            slope = slope.repeat(1, N, 1)

            eps = 1e-3
            x_shifts = eps * torch.cumsum(invalid_elements_mask, dim=1)
            y_shifts = slope * x_shifts
            Xslices_modified = Xslices_modified + y_shifts

            Xslices_sorted, _ = torch.sort(Xslices_modified, dim=1)

            x = torch.arange(1, N + 1).to(X.device) / (
                1 + torch.sum(mask, dim=1, keepdim=True)
            )  # B x N

            invalid_elements_mask = ~mask.bool()
            x_copy = x.clone()
            x_copy[invalid_elements_mask] = -1e10
            max_x, _ = torch.max(x_copy, dim=1, keepdim=True)
            x[invalid_elements_mask] = max_x.repeat(1, N)[invalid_elements_mask]

            x = x.unsqueeze(1).repeat(1, self.num_slices, 1) + torch.transpose(
                x_shifts, 1, 2
            )
            x = x.view(-1, N)  # BL x N

            xnew = (
                torch.linspace(0, 1, M + 2)[1:-1]
                .unsqueeze(0)
                .repeat(B * self.num_slices, 1)
                .to(X.device)
            )
            y = torch.transpose(Xslices_sorted, 1, 2).reshape(B * self.num_slices, -1)
            Xslices_sorted_interpolated = torch.transpose(
                Interp1d()(x, y, xnew).view(B, self.num_slices, -1), 1, 2
            )

        Rslices = self.reference.expand(Xslices_sorted_interpolated.shape)

        _, Rind = torch.sort(Rslices, dim=1)
        embeddings = (
            Rslices - torch.gather(Xslices_sorted_interpolated, dim=1, index=Rind)
        ).permute(0, 2, 1)  # B x num_slices x M

        weighted_embeddings = self.weight(embeddings).sum(-1)

        return weighted_embeddings.view(-1, self.num_slices)

    def get_slice(self, X):
        """
        Slices samples from distribution X~P_X
        Input:
            X:  B x N x dn tensor, containing a batch of B sets, each containing N samples in a dn-dimensional space
        """
        return self.theta(X)


# ** Basic autoencoder


class AePooling(BaseNN):
    def __init__(
        self,
        out_features: int | None = None,
        x_key: str = "x",
        y_key: str = "original",
        encoder_depth: int = 2,
        decoder_depth: int = 2,
        activation: nn.Module = nn.ReLU,
        conf: ModuleConfig | None = None,
    ) -> None:
        super().__init__(in_features=0, x_key=x_key, cfg=conf)
        self.task_names = [y_key]
        # self.pooling_weights: nn.Parameter = nn.Parameter(torch.)
        # TODO: probably want to review what's good here
        self.encoder: nn.ModuleList = nn.ModuleList()
        self.decoder: nn.ModuleList = nn.ModuleList()
        for _ in range(encoder_depth):
            self.encoder.append(nn.LazyLinear(out_features=out_features))
            self.encoder.append(activation())

        for _ in range(decoder_depth):
            self.decoder.append(nn.LazyLinear(out_features=out_features))
            self.decoder.append(activation())

    @override
    def forward(self, X: Tensor) -> Tensor:
        # Batched has shape (batch, n_seqs, embedding_dim)
        encoded: Tensor = self.encoder(X)
        return encoded.mean(axis=1)
        # Or should it be
        # return torch.einsum("abc,bc->ac", batch, self.pooling_weights)

    def criterion(
        self,
        y_pred,
        y_true,
        context: str | None = None,
        batch: dict | None = None,
        **kwargs,
    ):
        as_before = torch.nn.functional.pad(
            y_pred.unsqueeze(1), (0, 0, 0, self.embedding_size), value=1
        )
        decoded = self.pooler.decoder(as_before)
        return nn.functional.mse_loss(input=decoded, target=batch[self.x_key])
