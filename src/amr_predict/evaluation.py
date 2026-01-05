#!/usr/bin/env ipython

from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Literal, TypeAlias

import lightning as L
import numpy as np
import plotnine as gg
import polars as pl
import sklearn.model_selection as ms
import torch
import torch.nn as nn
from amr_predict.metrics import (
    multitask_all_cls,
    multitask_all_reg,
    multitask_metrics2df,
)
from amr_predict.models import Baseline
from amr_predict.utils import TASK_TYPES, Preprocessor, load_as
from datasets import Dataset, DatasetDict
from loguru import logger
from sklearn.preprocessing import LabelBinarizer
from torch import Tensor
from torch.utils.data import DataLoader

MODEL_CLASSES: TypeAlias = L.LightningModule | Baseline | nn.Module

logger.disable("amr_predict")


class Evaluator:
    def __init__(
        self,
        model: MODEL_CLASSES,
        how: Literal["cv", "holdout"] = "cv",
        preprocessor: Preprocessor | None = None,
        trainer: L.Trainer | None = None,
        **kws,
    ) -> None:
        self.model: MODEL_CLASSES = model
        self.pp: Preprocessor | None = preprocessor
        self.x_key: str = self.model.x_key
        self.task_type: TASK_TYPES = self.model.conf.task_type
        self.trainer: L.Trainer | None = trainer
        self.how: str = how
        self.kws: dict = kws

    def _fit(self, train: Dataset, val: Dataset | None = None) -> None:
        train = train.select_columns([self.model.x_key] + list(self.model.task_names))
        if isinstance(self.model, Baseline):
            logger.info("Start fit for Baseline model")
            self.model.fit(train)
            logger.success("Fit complete")
        elif self.trainer is None:
            raise ValueError("Trainer must be provided if not using baseline model")
        else:
            logger.info(f"Start fit for model {self.model}")
            tl = DataLoader(train, **self.kws)
            vl = DataLoader(val, **self.kws) if val is not None else val
            self.trainer.fit(self.model, train_dataloaders=tl, val_dataloaders=vl)
            logger.success("Fit complete")

    def cv(
        self,
        dataset: Dataset,
        validation_kws: dict | None = None,
        stratify_by: str | None = None,
        **kws,
    ) -> pl.DataFrame:
        """
        K-fold cross-validation
        [2025-10-16 Thu] WARNING: datasets.select might create new datasets, but you checked locally
        and it seemed not to

        Parameters
        ----------
        validation_kws : dict
            optional, keyword arguments for the proportion of the dataset to keep for validation
        stratify_by : str | None
            for classification tasks, the name of the task (column in dataset) to stratify
        kws : dict
            kws passed to the sklearn k-fold init
        """
        validation_kws = validation_kws or {}
        if not validation_kws:
            val_split = dataset.train_test_split(**validation_kws)
            val: Dataset | None = val_split["test"]
            dataset = val_split["train"]
        else:
            val = None
        if self.task_type == "classification" and stratify_by:
            y = dataset[stratify_by][:]
            k_fold = ms.RepeatedStratifiedKFold(**kws)
            splits = k_fold.split(np.zeros_like(dataset[self.x_key][:]), y=y)
        else:
            k_fold = ms.RepeatedKFold(**kws)
            splits = k_fold.split(np.zeros_like(dataset[self.x_key][:]))
        split_indices, split_names = {}, {}
        for i, (train, test) in enumerate(splits):
            fname = f"fold_{i}"
            train_key, test_key = f"{fname}_train", f"{fname}_test"
            if validation_kws:
                split_names[fname] = (train_key, test_key, val)
            else:
                split_names[fname] = (train_key, test_key, None)
            split_indices[train_key] = train
            split_indices[test_key] = test

        ddict = make_splits(dataset=dataset, split_methods=split_indices)
        return self.holdout(ddict, split_names)

    def holdout(
        self,
        dataset: Path | DatasetDict | Dataset,
        splits: dict[str, tuple[str, str, str | None | Dataset]] | None = None,
        validation_kws: dict | None = None,
        **kws,
    ) -> pl.DataFrame:
        """Holdout evaluation on a dataset dict, possibly saved on disk

        Parameters
        ----------
        validation_kws : dict
            Dictionary of keyword arguments for generating a validation set from
            the TRAIN data in splits. The "test" split is interpreted as the validation
            split
        splits : Sequence of (train, test, validation) tuple names. Validation set is
            optional

        Returns
        -------
        DataFrame of evaluation results
        """
        results: list[pl.DataFrame] = []
        if splits is None and isinstance(dataset, Dataset):
            if "valid_size" in kws:
                tmp = dataset.train_test_split(test_size=kws["valid_size"])
                del kws["valid_size"]
                validation = tmp["test"]
                dataset = tmp["train"]
            else:
                validation = None
            print("Using automatic split")
            dataset = dataset.train_test_split(**kws)
            splits = {"auto": ("train", "test", validation)}

        tasks = self.model.task_names
        for key, (train, test, val) in splits.items():
            if isinstance(val, str) and isinstance(dataset, Path):
                val_dset: Dataset | None = load_as(dataset / val)
            elif isinstance(val, Dataset):
                val_dset = val
            else:
                val_dset = dataset.get(val)
            if isinstance(dataset, Path):
                train_dset: Dataset = load_as(dataset / train)
                test_dset: Dataset = load_as(dataset / test)
            else:  # Interpret train, test to be keys and dataset to be DatasetDict
                train_dset = dataset[train]
                test_dset = dataset[test]
            logger.info(f"Holdout on key {key}")
            logger.info(f"Train, test shape: {train_dset.shape}, {test_dset.shape}")

            if validation_kws is not None:
                val_split = train_dset.train_test_split(**validation_kws)
                train_dset = val_split["train"]
                val_dset = val_split["test"]

            if val_dset is not None:
                logger.info(f"Validation set shape: {val_dset.shape}")

            if self.pp is not None:
                train_dset = self.pp.fit_transform(train_dset)
                test_dset = self.pp.transform(test_dset)
                if val_dset is not None:
                    val_dset = self.pp.transform(val_dset)

            self._fit(train=train_dset, val=val_dset)
            y_true: Tensor = test_dset.to_polars().select(tasks).to_torch()
            if self.task_type == "regression":
                y_pred: Tensor | tuple = self.model.predict_step(test_dset)
                metrics = multitask_all_reg(y_pred, y_true, task_names=tasks)
            else:
                y_pred = self.model.predict_proba(test_dset)
                metrics = multitask_all_cls(
                    y_pred,
                    y_true,
                    n_classes=self.model.conf.n_classes,
                    task_names=tasks,
                )
            df = multitask_metrics2df(metrics)
            results.append(df.with_columns(pl.lit(key).alias("test_set")))
        return pl.concat(results)


def make_splits(
    dataset: Dataset, split_methods: dict[str, Callable | np.ndarray | float]
) -> DatasetDict:
    """Helper function for splitting a hf dataset into a dataset dict

    Parameters
    ----------
    split_methods : dict[str, Callable | np.ndarray]
        dict where keys are the names of the splits, and values are either
        1. A function compatible with dataset.filter
        2. An array of indices, or a boolean mask
        3. A float, in which case dataset.train_test_split is called and the train, test
            splits are saved by treating the key as a suffix
    """
    indices = np.array(range(len(dataset)))
    result: DatasetDict = DatasetDict()
    for key, spec in split_methods.items():
        if isinstance(spec, Callable):
            result[key] = dataset.filter(spec)
        elif isinstance(spec, float):
            split = dataset.train_test_split(spec)
            result[f"{key}_test"] = split["test"]
            result[f"{key}_train"] = split["train"]
        elif isinstance(spec, np.ndarray) and spec.dtype == bool:
            ranges = indices[spec]
            result[key] = dataset.select(ranges)
        else:
            result[key] = dataset.select(spec)
    return result


# * SAE metrics
def categorize_latents(
    activations: Tensor, dense_threshold: float = 1 / 10
) -> dict[str, list | pl.Series]:
    """Helper function to flag dead and dense latents by their activation values"""
    result = {}
    n_samples = activations.shape[0]
    idx = torch.arange(activations.shape[1])
    frac_active = (activations > 0).sum(dim=0) / n_samples
    result["dead"] = idx[frac_active == 0].tolist()
    result["dense"] = idx[frac_active > dense_threshold]
    rest = idx[(frac_active != 0) & (frac_active <= dense_threshold)]
    result["sparse"] = rest
    result["sparse_df"] = pl.DataFrame(
        activations[:, rest], schema=[f"latent_{i}" for i in rest]
    )
    return result


def plot_activation_density(
    activations: Tensor, latent_idx: int, obs: pl.DataFrame, label_cols: Sequence
) -> dict[str, gg.ggplot]:
    """Plot the activation distribution for a single latent `latent_idx`, showing
    the relationship between it and the label classes in `label_cols`

    Parameters
    ----------
    activations : Tensor
        Tensor of shape n_samples x d_sae containing SAE activations
    obs : DataFrame
        DataFrame aligned to activations, containing annotations about samples
    label_cols : Sequence
        columns of `obs` to derive labels from

    Returns
    -------
    Dictionary of plot objects, keyed by column in label_cols
    """

    def plot_one(label_col):
        selected = activations[:, latent_idx]
        frac_active = ((selected > 0).sum() / selected.shape[0]) * 100
        to_df = pl.DataFrame({"Activation": selected, label_col: obs[label_col]})
        title = f"Latent {latent_idx}"
        return (
            gg.ggplot(to_df, gg.aes(x="Activation", fill=label_col))
            + gg.geom_histogram(binwidth=0.1)
            + gg.ggtitle(title, subtitle=f"Activation density: {frac_active}%")
            + gg.theme(title=gg.element_text(style="oblique"))
        )

    return {col: plot_one(col) for col in label_cols}


def highest_activations(
    activations: Tensor,
    obs: pl.DataFrame,
    label_col: str,
    k: int = 5,
    top_only: bool = True,
) -> dict[str, pl.DataFrame]:
    """Return the k latents that have the highest activations for each label in `label_col`

    Parameters
    ----------
    activations : Tensor
    obs : pl.DataFrame
        DataFrame aligned to activations
    top_only : bool
        In each dataframe, return information only about latents that showed up at least
        once in the top k for that label

    Returns
    -------
    A dictionary keyed by label, mapping to a dataframe with statistics about the top latents
        for that label.
    The dataframe has 3 columns: latent, fraction_top, fraction_active, max
    latent: the index of the latent
    fraction_top: the fraction of samples at which the latent appears in the top k
    fraction_active: the fraction of samples where the latent is nonzero
    max, median, mean: the max, median, and mean activation value of the latent
    """
    result = {}
    for label, group in obs.with_row_index().group_by(label_col):
        current: Tensor = activations[group["index"], :]
        n_samples = group.height
        active_frac = (current > 0).sum(dim=0) / n_samples
        latent_max = current.max(dim=0).values
        latent_median = current.median(dim=0).values
        latent_mean = current.mean(dim=0)
        top = current.topk(k=k, dim=1)
        top_counts = top.indices.flatten().unique(return_counts=True)
        top_frac = pl.DataFrame(
            {"latent": top_counts[0], "fraction_top": top_counts[1] / n_samples}
        )
        df: pl.DataFrame = (
            pl.DataFrame(
                {
                    "max": latent_max,
                    "mean": latent_mean,
                    "median": latent_median,
                    "fraction_active": active_frac,
                }
            )
            .with_row_index("latent")
            .join(top_frac, on="latent", how="inner")
            .sort("mean", descending=True)
        )
        if top_only:
            df = df.filter(pl.col("latent").is_in(top_counts[0]))
        result[label[0]] = df
    return result


def score_latents(activations: Tensor, labels: Sequence) -> pl.DataFrame:
    """Identify the best (most interpretable and monosemantic) latents in `activations`

    Parameters
    ----------
    activations : Tensor
        Tensor of SAE activations, of shape n_samples x d_sae
    labels : Sequence
        Labels for each sample in `activations`

    Returns
    -------
    Dataframe with the following columns:

    latent_idx: latent index
    label_max: label that the latent has the highest average activation for
    max_activation_avg: average of highest activation for `label_max`
    max_activation: the highest activation observed for `label_max`
    max_activation_prop: the proportion of `label_max`'s activations
        across the samples, out of the total average activation of the latent on all samples.
        A perfectly monosemantic latent should only fire for one label so
        the proportion should be one
        Averaging should alleviate issues of label imbalance

        The main metric for scoring latents, as it takes into account latents' activations
        on other labels

    Classification metrics, with respect to label_max, computed
        with one-vs-rest.

    Notes
    -----
    The resulting top latents are equivalent to those found with a contrastive approach
    that involves subtracting average activation values for each latent

    In each row, classification metrics for the given latent are defined with respect to
    the current `label_max` and against all other labels.
        so true positives are the count of samples with `label_max` where the latent had
        nonzero activations, true negatives are the sum of zero activations for all other
        labels and so on.
    """
    lidx = torch.arange(activations.shape[1])
    tmp: dict = {"latent_idx": lidx}
    encoder = LabelBinarizer()
    dtype = torch.get_default_dtype()
    ones: Tensor = torch.tensor(encoder.fit_transform(labels)).to(dtype)
    avg_acts = torch.t(activations).matmul(ones) / ones.sum(dim=0)
    maxes, idx = avg_acts.max(dim=1)
    tmp["max_activation_avg"] = maxes
    tmp["max_activation_prop"] = (maxes / avg_acts.sum(dim=1)).tolist()
    tmp["label_max"] = encoder.classes_[idx.numpy()]

    # t(activations) has shape d_sae x n
    # ones  has shape n x k
    # expands to d_sae x n x 1
    # and       1 x n x k
    a_expanded = torch.t(activations).unsqueeze(2)
    mvals, _ = torch.where(ones == 1, a_expanded, -torch.inf).max(dim=1)
    tmp["max_activation"] = mvals[lidx, idx].tolist()

    # Classifier scoring
    nonzero_counts = torch.t(activations > 0).to(dtype).matmul(ones)
    zero_counts = torch.t(activations == 0).to(dtype).matmul(ones)

    tp = nonzero_counts[lidx, idx]
    fn = zero_counts[lidx, idx]
    tn = zero_counts.sum(dim=1) - fn
    fp = nonzero_counts.sum(dim=1) - tp
    tmp["f1"] = ((2 * tp) / (2 * tp + fp + fn)).tolist()
    tmp["precision"] = tp / (tp + fp)
    tmp["fpr"] = fp / (fp + tn)
    tmp["fnr"] = fn / (fn + tp)
    tmp["sensitivity"] = tp / (tp + fn)
    tmp["specificity"] = tn / (tn + fp)

    return pl.DataFrame(tmp)
