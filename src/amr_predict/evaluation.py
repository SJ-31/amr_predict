#!/usr/bin/env ipython

from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Literal, TypeAlias

import jaxtyping
import lightning as L
import numpy as np
import PAMI.extras.dbStats.TransactionalDatabase as pstats
import pandera.polars as pa
import plotnine as gg
import polars as pl
import polars.selectors as cs
import sklearn.model_selection as ms
import torch
import torch.nn as nn
import umap
import umap.plot
from amr_predict.metrics import (
    multitask_all_cls,
    multitask_all_reg,
    multitask_metrics2df,
)
from amr_predict.models import Baseline
from amr_predict.utils import TASK_TYPES, Preprocessor, load_as, read_tabular
from attrs import Factory, define, field, fields_dict, validators
from beartype import beartype
from datasets import Dataset, DatasetDict
from loguru import logger
from matplotlib.axes import Axes
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from torch import Tensor
from torch.utils.data import DataLoader

MODEL_CLASSES: TypeAlias = L.LightningModule | Baseline | nn.Module

logger.disable("amr_predict")


TENSOR2D_FLOAT = jaxtyping.Float[Tensor, "a b"]


class Evaluator:
    def __init__(
        self,
        model: MODEL_CLASSES,
        preprocessor: Preprocessor | None = None,
        trainer: L.Trainer | None = None,
        model_fn: Callable[[Any], MODEL_CLASSES] | None = None,
        **kws,
    ) -> None:
        """
        Parameters
        ----------
        model_fn : Callable
            Function that takes a single argument: the number of input features (as integer) and
            returns a trainable model. Required if the preprocessor is expected to change
            the number of features and `model` needs to be re-instantiated to handle the change
        """
        self.model: MODEL_CLASSES = model
        self.pp: Preprocessor | None = preprocessor
        self.model_fn: Callable[[Any], MODEL_CLASSES] | None = model_fn
        self.x_key: str = self.model.x_key
        self.task_type: TASK_TYPES = self.model.cfg.task_type
        self.trainer: L.Trainer | None = trainer
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
        splits: dict[str, list[str | None | Dataset]] | None = None,
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
            logger.info("Using automatic split")
            dataset = dataset.train_test_split(**kws)
            splits = {"auto": ("train", "test", validation)}

        tasks = self.model.task_names
        for key, dsets in splits.items():
            val = None
            if len(dsets) == 2:
                train, test = dsets
            elif len(dsets) == 3:
                train, test, val = dsets
            else:
                raise ValueError(
                    "At least train, test datasets must be supplied for holdout"
                )
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
            if test_dset.shape[0] == 0:
                raise ValueError("no samples in test dataset")
            if train_dset.shape[0] == 0:
                raise ValueError("no samples in train dataset")

            if validation_kws:
                val_split = train_dset.train_test_split(**validation_kws)
                logger.info("Generating validation set from kws {}", validation_kws)
                train_dset = val_split["train"]
                val_dset = val_split["test"]
            if val_dset is not None:
                logger.info(f"Validation set shape: {val_dset.shape}")

            if self.pp is not None:
                train_dset, test_dset, val_dset = self._preprocess(
                    train_dset, test_dset, val_dset
                )
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
                    n_classes=self.model.cfg.n_classes,
                    task_names=tasks,
                )
            df = multitask_metrics2df(metrics)
            results.append(df.with_columns(pl.lit(key).alias("test_set")))
        if not results:
            raise ValueError("no splits were given")
        return pl.concat(results)

    def _preprocess(self, train_dset, test_dset, val_dset):
        old_in_features: int = train_dset[self.x_key][:].shape[1]
        train_dset = self.pp.fit_transform(train_dset)
        test_dset = self.pp.transform(test_dset)
        new_in_features: int = train_dset[self.x_key][:].shape[1]
        if new_in_features != test_dset[self.x_key][:].shape[1]:
            raise ValueError(
                "Number of features in train and test datasets must be identical after preprocessing"
            )
        if val_dset is not None:
            val_dset = self.pp.transform(val_dset)
            if new_in_features != val_dset[self.x_key][:].shape[1]:
                raise ValueError(
                    "Number of features in train and validation datasets must be identical after preprocessing"
                )
        if new_in_features != old_in_features:
            logger.warning("""
            Number of features changed during preprocessing.
            Ensure the model can handle this if no `model_fn` was passed
            """)
        if self.model_fn is not None:
            self.model = self.model_fn(new_in_features)
        return train_dset, test_dset, val_dset


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


def make_control_task(
    data: pl.DataFrame | Dataset,
    target_task: str,
    control_col: str,
    seed: int | None = None,
    add: bool = False,
    added_name: str | None = None,
) -> dict | pl.DataFrame | Dataset:
    rng = np.random.default_rng(seed)
    was_torch: bool = False
    if isinstance(tt := data[target_task][:], Tensor):
        targets = list(set(tt.tolist()))
        was_torch = True
    else:
        targets = list(set(tt))
    if isinstance(ct := data[control_col][:], Tensor):
        control_labels = list(set(ct.tolist()))
        was_torch = True
    else:
        control_labels = list(set(ct))
    rng.shuffle(targets)
    rng.shuffle(control_labels)
    if len(control_labels) < len(targets):
        logger.debug("targets: {}", targets)
        logger.debug("control labels: {}", control_labels)
        raise ValueError(
            "The number of unique control labels needs to be at least equal to the number of target labels"
        )
    mapping: dict = {}
    # Map at least one of control labels to targets
    for t in targets:
        chosen = rng.choice(control_labels, 1, replace=False).item()
        mapping[chosen] = t
        control_labels.remove(chosen)
    # If any control labels remain, keep adding mappings
    while control_labels:
        cl = rng.choice(control_labels, 1, replace=False).item()
        mapping[cl] = rng.choice(targets, 1, replace=False).item()
        control_labels.remove(cl)
    if not add:
        return mapping
    new_col = added_name or f"control_task_{target_task}-{control_col}"
    if isinstance(data, pl.DataFrame):
        return data.with_columns(pl.col(control_col).replace(mapping).alias(new_col))

    def replace(batch):
        batch[new_col] = [mapping[ctrl] for ctrl in batch[control_col]]
        return batch

    mapped = data.map(replace, batched=True, batch_size=data.shape[0])
    if was_torch:
        return mapped.with_format("torch")
    return mapped


# * SAE metrics


@define
class SaeMetrics:
    lidx: pl.Series = field(
        converter=lambda val: val if isinstance(val, pl.Series) else pl.Series(val)
    )
    labels: pl.Series = field(
        converter=lambda val: val if isinstance(val, pl.Series) else pl.Series(val)
    )
    sensitivity: Tensor = field(validator=validators.instance_of(TENSOR2D_FLOAT))
    fpr: Tensor = field(validator=validators.instance_of(TENSOR2D_FLOAT))
    fnr: Tensor = field(validator=validators.instance_of(TENSOR2D_FLOAT))
    specificity: Tensor = field(validator=validators.instance_of(TENSOR2D_FLOAT))
    precision: Tensor = field(validator=validators.instance_of(TENSOR2D_FLOAT))
    negative_predictive_value: Tensor = field(
        validator=validators.instance_of(TENSOR2D_FLOAT)
    )
    accuracy: Tensor = field(validator=validators.instance_of(TENSOR2D_FLOAT))
    activation_prop: Tensor = field(validator=validators.instance_of(TENSOR2D_FLOAT))

    def report(self, k: int = 1, by: str = "activation_prop") -> pl.DataFrame:
        """Produce a dataframe which for each latent, reports the top n labels for each
        metric

        Parameters
        ----------
        n : int
            Number of labels to report
        by : str
            The metric by which to rank label scores for the latent

        Returns
        -------
        Dataframe with the following columns:

        latent_idx: latent index
        activation_prop: the proportion of `label`'s activations
            across the samples, out of the total average activation of the latent on all samples.
            A perfectly monosemantic latent should only fire for one label so
            the proportion should be one
            Averaging should alleviate issues of label imbalance

        Classification metrics, with respect to label, computed
            with one-vs-rest.

        Notes
        -----
        The top latents defined by `activation_prop`
        are equivalent to those found with a contrastive approach
        that involves subtracting average activation values for each latent

        In each row, classification metrics for the given latent are defined with respect to
        the current `label` against all other labels.
            so true positives are the count of samples with `label` where the latent had
            nonzero activations, true negatives are the sum of zero activations for all other
            labels and so on.

        If k > 1, the results for the top k labels are returned in lists
        """
        metric_fields = [
            f for f in fields_dict(SaeMetrics).keys() if f not in {"labels", "lidx"}
        ]
        assert by in metric_fields, f"`by` must be one of {metric_fields} "
        data: Tensor = getattr(self, by)
        topk_vals, topk_idx = data.topk(k=k, dim=0)
        tmp = {"latent_idx": self.lidx, by: topk_vals.transpose(0, 1)}
        topk_idx = topk_idx.numpy()
        for metric in metric_fields:
            if metric != by:
                cur = getattr(self, metric).numpy()
                tmp[metric] = [cur[topk_idx[:, i], i] for i in range(len(self.lidx))]
        tmp["label"] = [self.labels[topk_idx[:, i]] for i in range(len(self.lidx))]
        tmp["frac_active"] = (self.activation_prop > 0).sum(
            dim=0
        ) / self.activation_prop.shape[0]
        if k != 1:
            return pl.DataFrame(tmp)
        return pl.DataFrame(tmp).with_columns(
            cs.array().arr.first(), cs.list().list.first()
        )


@define
class EvalSAE:
    acts: Tensor = field(validator=lambda _, __, value: len(value.shape) == 2)
    threshold: float
    # Activations of shape n_samples x d_sae
    U: umap.UMAP = field(init=False)
    acts_grouped: pl.DataFrame = field(init=False)
    categories: dict = field(factory=dict, init=False)
    lidx: pl.Series = field(
        init=False,
        default=Factory(
            lambda self: pl.Series([f"l{i}" for i in range(self.acts.shape[1])]),
            takes_self=True,
        ),
    )
    acts_dtype: torch.dtype = field(
        init=False, default=Factory(lambda self: self.acts.dtype, takes_self=True)
    )
    n: int = field(
        init=False, default=Factory(lambda self: self.acts.shape[0], takes_self=True)
    )

    def drop_latents(
        self,
        drop_dead: bool = True,
        drop_dense: bool = False,
        include_samples: bool = True,
        inplace: bool = False,
        **kws,
    ) -> pl.DataFrame | None:
        """Drop dead and/or dense latents from the activation matrix
        Optionally, drop dead samples as well i.e. zero rows
        """
        cats: dict = self.categories or self.categorize_latents(**kws)
        dropped = pl.DataFrame(self.acts, schema=self.lidx.to_list())
        if drop_dead:
            dropped = dropped.drop(cats["dead"]["latent_idx"].to_list())
        if drop_dense:
            dropped = dropped.drop(cats["dense"]["latent_idx"].to_list())
        if include_samples:
            dropped = dropped.filter(
                ~pl.all_horizontal(pl.all() <= kws.get("active_threshold", 0))
            )
        dropped = dropped.select(dropped.columns)
        if not inplace:
            return dropped.to_torch().to(self.acts_dtype)
        self.acts = dropped.to_torch().to(self.acts_dtype)

    def categorize_latents(
        self,
        dense_threshold: float = 1 / 10,
        save: bool = False,
        active_threshold: float = 0.0,
    ) -> dict[str, list | pl.Series]:
        """Helper function to flag dead and dense latents by their activation values"""
        result = {}
        frac_active = pl.DataFrame(
            (self.acts >= active_threshold).sum(dim=0).reshape((1, -1)) / self.n,
            schema=self.lidx.to_list(),
        ).unpivot(variable_name="latent_idx", value_name="frac_active")
        result["dead"] = frac_active.filter(pl.col("frac_active") <= active_threshold)
        result["dense"] = frac_active.filter(pl.col("frac_active") > dense_threshold)
        rest = frac_active.filter(
            (pl.col("frac_active") > active_threshold)
            & (pl.col("frac_active") <= dense_threshold)
        )
        result["sparse"] = rest
        if save:
            self.categories.update(result)
        return result

    def plot_activation_density(
        self,
        latent_idx: str | Sequence,
        obs: pl.DataFrame,
        label_cols: Sequence,
        bins: int = 15,
        binwidth=0.1,
        log_x: bool = True,
        top_labels: dict[str, Sequence] | None = None,
        ncol: int | None = None,
        nrow: int | None = None,
    ) -> dict[str, tuple[gg.ggplot, pl.DataFrame]] | tuple[gg.ggplot, pl.DataFrame]:
        """Plot the activation distribution for a single latent `latent_idx`, showing
        the relationship between it and the label classes in `label_cols`

        Parameters
        ----------
        obs : DataFrame
            DataFrame aligned to activations, containing annotations about samples
        label_cols : Sequence
            columns of `obs` to derive labels from
        top_labels : dict[(str, str), str]
            dictionary returning values from `label_cols` to display as the top label for each latent
            e.g. obtained from the `label_max` column with `score_latents`.
            Keys are tuples of (latent_idx, label_col)

        Returns
        -------
        Dictionary of plot objects, keyed by column in label_cols
        """
        df: pl.DataFrame
        frac_active: float | None = None
        latent2fa: dict[str, float] = {}
        if isinstance(latent_idx, str):
            only_one = True
            df, frac_active = self._get_activation_df(latent_idx)
        else:
            only_one = False
            dfs = []
            for idx in latent_idx:
                cur, frac = self._get_activation_df(idx)
                cur = cur.with_columns(pl.lit(idx).alias("Latent id")).with_row_index()
                latent2fa[idx] = str(round(frac, 1))
                dfs.append(cur)
            df = pl.concat(dfs)

        def labeller(colname: str, label_col: str) -> str:  # colname is the latent_idx
            val = f"{colname} ({latent2fa[colname]}%)"
            if top_labels is not None and (colname, label_col) in top_labels:
                val = f"{val}\nTop label: {top_labels[(colname, label_col)]}"
            return val

        def plot_one(label_col):
            if only_one:
                tmp = df.with_columns(pl.Series(obs[label_col]).alias(label_col))
                title = f"Latent {latent_idx}"
                subtitle = f"Activation density: {round(frac_active, 1)}%"
                if top_labels is not None:
                    subtitle = (
                        f"{subtitle}, Top label: {top_labels[(latent_idx, label_col)]}"
                    )
            else:
                tmp = df.join(
                    pl.Series(obs[label_col]).to_frame(label_col).with_row_index(),
                    on="index",
                    how="left",
                    validate="m:1",
                )
                title = "Latent Activations"
                subtitle = "Subplot name format: <latent id> (<total active>%)"
            plot = (
                gg.ggplot(tmp, gg.aes(x="Activation", fill=label_col))
                + gg.geom_histogram(binwidth=binwidth, bins=bins)
                + gg.ggtitle(title, subtitle=subtitle)
                + gg.theme(plot_title=gg.element_text(style="oblique"))
                + gg.xlab("Activation value")
            )
            if not only_one:
                plot = plot + gg.facet_wrap(
                    "Latent id",
                    scales="fixed",
                    labeller=gg.labeller(cols=lambda x: labeller(x, label_col)),
                    ncol=ncol,
                    nrow=nrow,
                )
            if log_x:
                plot = (
                    plot
                    + gg.scale_x_log10()
                    + gg.theme(panel_grid_minor_x=gg.element_blank())
                )
            return plot, tmp

        if len(label_cols) == 1:
            return plot_one(label_cols[0])
        return {col: plot_one(col) for col in label_cols}

    def _get_activation_df(self, idx: str) -> tuple[pl.DataFrame, float]:
        selected = self.acts[idx]
        frac_active: float = ((selected > 0).sum() / selected.shape[0]) * 100
        return pl.DataFrame(({"Activation": selected})), frac_active

    def umap(
        self, from_grouped: bool, return_array: bool = False, **kws
    ) -> None | np.ndarray:
        if from_grouped:
            arr = self.acts_grouped.drop("latent_idx").to_numpy()
        else:
            arr = self.acts.to_numpy().transpose()
        self.U = umap.UMAP(**kws)
        if return_array:
            return self.U.fit_transform(arr)
        self.U.fit(arr)

    def plot_umap(self, labels: Sequence | None = None, **kws) -> Axes | Any:
        if self.U is not None:
            return umap.plot.points(self.U, labels=labels, **kws)
        raise ValueError("UMAP object not found, run self.umap first")

    def cluster_latents(
        self,
        from_grouped: bool,
        cluster_obj: Callable = KMeans,
        labels: Sequence | None = None,
        silhouette: bool = True,
        ch_index: bool = True,
        **kws,
    ) -> pl.DataFrame:
        """
        Cluster latents by their activation values across samples

        Parameters
        ----------
        from_grouped : bool
            Whether to use previously grouped latents
        cluster_obj : Callable
            Sklearn-style clustering class. Needs `fit_predict` method that produces clusters
        label_n_clusters : bool
            Whether to supply the number of unique labels as a parameter "n_clusters"
        kws : Keyword arguments
            Passed to cluster_obj init
        Returns
        -------
        A DataFrame describing latent cluster assignments and possibly clustering metrics
        """
        if from_grouped and self.acts_grouped is not None:
            acts: np.ndarray = self.acts_grouped.drop("latent_idx").to_numpy()
            lnames = self.acts_grouped["latent_idx"]
        elif from_grouped:
            raise ValueError("Must call `group_by_labels` first to use `from_grouped`")
        else:
            acts = self.acts.to_numpy().transpose()
            lnames = self.acts.columns
        # `acts` has shape feature_size x n_samples
        if labels:
            kws["n_clusters"] = len(set(labels))
        clst = cluster_obj(**kws)
        assignments = clst.fit_predict(acts)
        n_samples = acts.shape[0]
        tmp = {"latent_idx": lnames, "cluster": assignments}
        if silhouette:
            tmp["silhouette_samples"] = silhouette_samples(acts, assignments)
            tmp["silhouette_score"] = [silhouette_score(acts, assignments)] * n_samples
        if ch_index:
            tmp["ch_index"] = [calinski_harabasz_score(acts, assignments)] * n_samples
        return pl.DataFrame(tmp)

    def group_by_labels(
        # TODO: [2026-04-17 Fri] probably don't need this now
        self,
        labels: Sequence,
        agg: Literal["mean", "max", "sum"] = "mean",
    ) -> None:
        if agg == "max":
            self.acts_grouped = max_by_label(self.acts, labels).with_columns(
                pl.Series(self.acts.columns).alias("latent_idx")
            )
            return
        ones, _, unique_labels = encode_labels(labels)
        ones = torch.tensor(ones).to(self.acts_dtype)
        grouped = torch.matmul(self.acts.transpose(0, 1), ones)
        if agg == "mean":
            grouped = grouped / ones.sum(dim=0)
        self.acts_grouped = pl.DataFrame(
            grouped, schema=list(unique_labels)
        ).with_columns(pl.Series(self.lidx).alias("latent_idx"))

    def _compute_metrics(
        self,
        anno_occurence: Tensor,
        activation_prop: Tensor,
        labels: pl.Series,
    ) -> SaeMetrics:
        """
        Calculate standard classification metrics for each latent's activation on the
        a series of concepts.

        Parameters
        ----------
        anno_occurence : Tensor
            Binary matrix of shape n_concepts x n_samples, where the i,j entry is
            1 if sammple j has concept i
        """
        pred_active = (self.acts >= self.threshold).to(self.acts_dtype)
        pred_dead = torch.where(pred_active == 1, 0, 1).to(self.acts_dtype)
        anno_inverted = torch.where(anno_occurence == 1, 0, 1).to(self.acts_dtype)

        # All matrices below of shape G x dim_size
        true_pos = torch.matmul(anno_occurence, pred_active)
        false_pos = torch.matmul(anno_inverted, pred_active)
        false_neg = torch.matmul(anno_occurence, pred_dead)
        true_neg = torch.matmul(anno_inverted, pred_dead)

        return SaeMetrics(
            lidx=self.lidx,
            labels=labels,
            sensitivity=true_pos / (true_pos + false_neg),
            specificity=true_neg / (true_neg + false_pos),
            fnr=false_neg / (false_neg + true_pos),
            fpr=false_pos / (false_pos + true_neg),
            precision=true_pos / (true_pos + false_pos),
            accuracy=(true_pos + true_neg) / self.n,
            negative_predictive_value=true_neg / (true_neg + false_neg),
            activation_prop=activation_prop,
        )

    def score_latents(
        self,
        labels: pl.DataFrame,
        label_col: str,
        sample_col: str = "sample",
        label_sep: str = ";",
    ) -> SaeMetrics:
        """Score SAE latents for samples annotated with multiple labels
        i.e. labels that aren't mutually exclusive

        Parameters
        ----------
        labels : pl.DataFrame
            DataFrame with samples as rows
        label_col : str
            String column in `labels` containing 0 or more labels for the sample,
            delimited by `label_sep`
        """
        occurrences: pl.LazyFrame = to_binary_form(
            labels, label_col=label_col, sample_col=sample_col, sep=label_sep
        ).lazy()
        occur_vals: Tensor = (
            occurrences.collect().transpose().to_torch().to(self.acts_dtype)
        )
        sum_acts = torch.matmul(occur_vals, self.acts)
        return self._compute_metrics(
            anno_occurence=occur_vals,
            labels=occurrences.collect_schema().names(),
            activation_prop=sum_acts / sum_acts.sum(dim=0),
        )


def to_binary_form(
    df: pl.DataFrame, sample_col: str, label_col: str, sep: str = ";"
) -> pl.DataFrame:
    result = (
        df.select(sample_col, label_col)
        .with_columns(pl.col(label_col).str.split(sep))
        .explode(label_col)
        .with_columns(pl.lit(1).alias("val"))
        .pivot(
            label_col,
            index=sample_col,
            values="val",
            aggregate_function="first",
        )
        .fill_null(0)
        .drop(sample_col)
    )
    return result


# * Utility functions


def encode_labels(labels) -> tuple[np.ndarray, LabelBinarizer | OneHotEncoder, list]:
    """Encode labels to to produce a matrix, even for binary labels"""
    if len(set(labels)) > 2:
        encoder = LabelBinarizer()
        label_mat = encoder.fit_transform(labels)
        labs = encoder.classes_
    else:
        encoder = OneHotEncoder(sparse_output=False)
        label_mat = encoder.fit_transform(np.array(labels).reshape(-1, 1))
        labs = encoder.categories_[0]
    return label_mat, encoder, labs


def max_by_label(
    activations: Tensor | pl.DataFrame, labels: Sequence
) -> Tensor | pl.DataFrame:
    """Return a matrix of (activations.shape[1], n_labels)
    where each entry is the maximum value of `activations` within labels
    """
    if activations.shape[0] != len(labels):
        raise ValueError("The number of activations and labels must match")
    label_mat, _, unique_labels = encode_labels(labels)
    if isinstance(activations, Tensor):
        dtype = torch.get_default_dtype()
        ones: Tensor = torch.tensor(label_mat).to(dtype)
        expanded = torch.t(activations).unsqueeze(2)
        mvals, _ = torch.where(ones == 1, expanded, -torch.inf).max(dim=1)
        return mvals
    else:
        labeled = activations.with_columns(pl.lit(labels).alias("label"))
        return (
            labeled.group_by("label", maintain_order=True)
            .agg(pl.all().max())
            .sort("label")
            .drop("label")
            .transpose()
            .rename(lambda x: x.removeprefix("column_"))
        )


def pami_wrapper(
    df: pl.DataFrame,
    obj,
    label_col: str = "labels",
    sep: str = ";",
    tmp_file: str | None = None,
    min_sup: int | float = 0.4,
    with_stats: bool = True,
) -> tuple[pl.DataFrame, dict | None]:
    write_to = tmp_file or NamedTemporaryFile("w").name
    df.select(label_col).write_csv(write_to, include_header=False)
    stats = None
    if with_stats:
        stats = {}
        tdb = pstats.TransactionalDatabase(write_to, sep=sep)
        tdb.run()
        for attr in [
            "getAverageTransactionLength",
            "getDatabaseSize",
            "getDensity",
            "getFrequenciesInRange",
            "getMaximumTransactionLength",
            "getMinimumTransactionLength",
            "getNumberOfItems",
            "getSortedListOfItemFrequencies",
            "getSparsity",
            "getStandardDeviationTransactionLength",
            "getTotalNumberOfItems",
            "getTransanctionalLengthDistribution",
            "getVarianceTransactionLength",
        ]:
            key = attr.replace("get", "")
            stats[key] = getattr(tdb, attr)()
    alg = obj(write_to, minSup=min_sup, sep=";")
    alg.mine()
    if tmp_file is not None:
        Path(write_to).unlink()
    return pl.from_pandas(alg.getPatternsAsDataFrame()).with_columns(
        pl.col("Patterns").str.split(sep),
        (pl.col("Support").cast(pl.Int64) / df.height).alias("Proportion"),
    ), stats


@define
class LabelCooccur:
    label_df: pl.DataFrame = field()

    @label_df.validator
    def _validate_ldf(self, _, value):
        schema = pa.DataFrameSchema(
            {
                self.label_col: pa.Column(pl.String),
                self.sample_col: pa.Column(unique=True),
            }
        )
        schema.validate(value)
        return (
            value.with_columns(pl.col(self.label_col).str.split(self.sep))
            .explode(self.label_col)
            .unique(self.label_col)[self.label_col]
            .is_in(set(self.sae_metrics.labels))
            .all()
        )

    sae_metrics: SaeMetrics
    label_col: str
    sample_col: str = "sample"
    sep: str = ";"
    max_fpr: float = 0.2

    @beartype
    def higher_order(
        self,
        k: int = 4,
        min_sup: int | float = 0.4,
        by: str = "activation_prop",
        tmp_file: str | None = None,
        p_overlap: float = 0.8,
        pami_previous: str | Path | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, dict | None]:
        """Use FPGrowth from the PAMI
        library to identify co-occuring labels, and compare with top latent
        activations

        Parameters
        ----------
        min_sup : int | float
            Minimum proportion or number of times a label set needs to occur in the data
            to be included
        k : int
            Number of top labels to include
        max_fpr : float
            Filter out latents with a FPR higher than this value on its top labels
        p_overlap : float
            The minimum percentage of overlap (intersection) between a
            latents' top labels and
            the frequent label set for it to be considered. Calculated with respect to
            the size of the label set
        pami_previous : str | None
            Path to a file containing the results from a previous run of PAMI. The "Patterns"
            column in this file are assumed to have the same separator as self.sep

        Returns
        -------
        Tuple of two dataframes and a dictionary of the FPGrowth statistics

        The first df is the result of joining top latents with frequent candidates
        """
        if k <= 2:
            raise ValueError("Use the `pairs` method for k == 2")
        if pami_previous is None:
            from PAMI.frequentPattern.basic.FPGrowth import FPGrowth

            cuda_apriori_available = False
            cudaAprioriTID = None
            try:
                from PAMI.frequentPattern.cuda import cudaAprioriTID

                cuda_apriori_available: bool = True
            except (ModuleNotFoundError, ImportError):
                pass
            if torch.cuda.is_available() and cuda_apriori_available:
                alg = cudaAprioriTID
            else:
                alg = FPGrowth
            frequent_patterns, pattern_stats = pami_wrapper(
                self.label_df,
                alg,
                label_col=self.label_col,
                sep=self.sep,
                min_sup=min_sup,
                tmp_file=tmp_file,
            )
        else:
            frequent_patterns = read_tabular(pami_previous)
            schema = pa.DataFrameSchema(
                {"Patterns": pa.Column(pl.String), "Support": pa.Column(pl.Int64)}
            )
            schema.validate(frequent_patterns)
            frequent_patterns = frequent_patterns.with_columns(
                pl.col("Patterns").str.split(self.sep)
            )
            pattern_stats = None

        top_latents = (
            self.sae_metrics.report(k=k, by=by)
            .with_columns(pl.col("label").list.sort())
            .filter(pl.col("fpr").arr.min() <= self.max_fpr)
            .lazy()
        )
        to_join = (
            frequent_patterns.lazy()
            .with_columns(
                pl.col("Patterns").list.sort(),
                pl.col("Patterns").list.len().alias("Size"),
            )
            .filter(pl.col("Size") >= 2)
        )
        joined = top_latents.join_where(
            to_join,
            (
                pl.col("label").list.set_intersection(pl.col("Patterns")).list.len()
                / pl.col("Size")
            )
            >= p_overlap,
        ).collect()
        return joined, frequent_patterns, pattern_stats

    def pairs(self, by: str = "activation_prop") -> pl.DataFrame:
        report = self.sae_metrics.report(k=2, by=by).filter(
            pl.col("fpr").arr.min() <= self.max_fpr
        )
        binary = to_binary_form(
            self.label_df,
            sample_col=self.sample_col,
            label_col=self.label_col,
            sep=self.sep,
        )
        labels = pl.Series(binary.columns)
        mat = torch.matmul(binary.to_torch().transpose(0, 1), binary.to_torch())
        report = (
            report.with_columns(
                pl.col("label")
                .map_elements(
                    lambda x: mat[labels.index_of(x[0]), labels.index_of(x[1])],
                    return_dtype=pl.Int64,
                )
                .alias("pair_coocurrence")
            )
            .with_columns(
                (pl.col("pair_coocurrence") / binary.height).alias(
                    "pair_cooccurrence_prop"
                )
            )
            .sort("pair_cooccurrence_prop", descending=True)
        )
        return report
