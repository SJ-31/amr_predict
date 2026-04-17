#!/usr/bin/env ipython

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias

import lightning as L
import numpy as np
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
from amr_predict.utils import TASK_TYPES, Preprocessor, load_as
from attrs import Factory, define, field, fields_dict
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
    mutually_exclusive: bool
    sensitivity: Tensor = field(validator=lambda _, __, value: len(value.shape) == 2)
    fpr: Tensor = field(validator=lambda _, __, value: len(value.shape) == 2)
    fnr: Tensor = field(validator=lambda _, __, value: len(value.shape) == 2)
    specificity: Tensor = field(validator=lambda _, __, value: len(value.shape) == 2)
    precision: Tensor = field(validator=lambda _, __, value: len(value.shape) == 2)
    negative_predictive_value: Tensor = field(
        validator=lambda _, __, value: len(value.shape) == 2
    )
    accuracy: Tensor = field(validator=lambda _, __, value: len(value.shape) == 2)
    activation_prop: Tensor = field(
        validator=lambda _, __, value: len(value.shape) == 2
    )

    def report(self, n: int = 1, by: str = "activation_prop") -> pl.DataFrame:
        """Produce a dataframe which for each latent, reports the top n labels for each
        metric

        Parameters
        ----------
        n : int
            Number of labels to report
        by : str
            The metric by which to rank label scores for the latent
        """
        metric_fields = [
            f
            for f in fields_dict(SaeMetrics).keys()
            if f not in {"labels", "lidx", "mutually_exclusive"}
        ]
        assert by in metric_fields, f"`by` must be one of {metric_fields} "
        data: Tensor = getattr(self, by)
        topk_vals, topk_idx = data.topk(k=n, dim=0)
        tmp = {"latent_idx": self.lidx, by: topk_vals.transpose(0, 1)}
        topk_idx = topk_idx.numpy()
        for metric in metric_fields:
            if metric != by:
                cur = getattr(self, metric).numpy()
                tmp[metric] = [cur[topk_idx[:, i], i] for i in range(len(self.lidx))]
        tmp["label"] = [self.labels[topk_idx[:, i]] for i in range(len(self.lidx))]
        return pl.DataFrame(tmp).with_columns(
            pl.when(n == 1).then(cs.array().arr.first()),
            pl.when(n == 1).then(cs.list().list.first()),
        )


@define
class EvalSAE:
    acts: Tensor = field(validator=lambda _, __, value: len(value.shape) == 2)
    threshold: float
    # Activations of shape n_samples x d_sae
    U: umap.UMAP = field(init=False)
    acts_grouped: pl.DataFrame = field(init=False)
    categories: dict = field(factory=dict, init=False)
    lidx: list = field(
        init=False,
        default=Factory(
            lambda self: [f"l{i}" for i in range(self.acts.shape[1])], takes_self=True
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
        dropped = self.acts
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
            return dropped
        self.acts = dropped

    def categorize_latents(
        self,
        dense_threshold: float = 1 / 10,
        save: bool = False,
        active_threshold: float = 0.0,
    ) -> dict[str, list | pl.Series]:
        """Helper function to flag dead and dense latents by their activation values"""
        result = {}
        frac_active = (
            (self.acts >= active_threshold).sum() / self.acts.height
        ).unpivot(variable_name="latent_idx", value_name="frac_active")
        result["dead"] = frac_active.filter(pl.col("frac_active") <= active_threshold)
        result["dense"] = frac_active.filter(pl.col("frac_active") > dense_threshold)
        rest = frac_active.filter(
            (pl.col("frac_active") > active_threshold)
            & (pl.col("frac_active") <= dense_threshold)
        )
        result["sparse"] = rest
        result["sparse_df"] = self.acts.select(rest["latent_idx"].to_list())
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

    def score_latents(self, labels: pl.DataFrame | Sequence, **kws):
        if isinstance(labels, pl.DataFrame):
            return self._score_with_multiple(labels=labels, **kws)
        return self._score_mutually_exclusive(labels)

    def _compute_multi_metrics(
        self,
        anno_occurence: Tensor,
        latent_type: Literal["firing", "dead"],
    ) -> dict:
        """
        Calculate standard classification metrics for each latent's activation on the
        a series of concepts.

        Parameters
        ----------
        anno_occurence : Tensor
            Binary matrix of shape n_concepts x n_samples, where the i,j entry is
            1 if sammple j has concept i
        """
        result = {}
        if latent_type == "firing":
            preds = self.acts >= self.threshold
        else:
            preds = self.acts < self.threshold
            anno_occurence = torch.where(anno_occurence == 1, 0, 1).to(self.acts_dtype)
        preds = preds.to(self.acts_dtype)
        correct_preds = torch.matmul(anno_occurence, preds)  # G x dim_size
        # When latent_type == "firing", correct_preds is the sum of true positives
        # otherwise it is the sum of true negatives
        pred_sum = preds.sum(dim=0).reshape((1, -1))
        truth = anno_occurence.sum(dim=1).reshape((-1, 1))
        # `truth` is true positives when "firing", else true negatives
        if latent_type == "firing":
            result["sensitivity"] = correct_preds / truth
            result["precision"] = correct_preds / pred_sum
            result["true_positives"] = correct_preds
        else:
            result["specificity"] = correct_preds / truth
            result["negative_predictive_value"] = correct_preds / pred_sum
            result["true_negatives"] = correct_preds / truth
        return result

    def _score_with_multiple(
        self,
        labels: pl.DataFrame,
        label_col: str,
        sample_col: str,
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
        occurrences: pl.LazyFrame = (
            labels.with_columns(pl.col(label_col).str.split(label_sep))
            .explode(label_col)
            .with_columns(pl.lit(1).alias("val"))
            .pivot(
                label_col,
                index=sample_col,
                values="val",
                aggregate_function="first",
            )
            .fill_null(0)
        ).lazy()
        occur_vals: Tensor = (
            occurrences.collect()
            .drop(sample_col)
            .transpose()
            .to_torch()
            .to(self.acts_dtype)
        )
        multiplied: pl.LazyFrame = (
            pl.DataFrame(torch.matmul(occur_vals, self.acts))
            .lazy()
            .with_columns(annotation=occurrences.drop(sample_col).columns)
        )
        tmp_results: dict = {}
        tmp_results["activation_prop"] = multiplied.with_columns(
            cs.exclude("anno") / cs.exclude("anno").sum()
        )
        for lt in ("firing", "dead"):
            tmp_results.update(
                self._compute_multi_metrics(anno_occurence=occur_vals, latent_type=lt)
            )
        tmp_results["accuracy"] = (
            tmp_results.pop("true_positives")
            + tmp_results.pop("true_negatives") / self.n
        )
        return SaeMetrics(**tmp_results)

    def _score_mutually_exclusive(self, labels: Sequence) -> pl.DataFrame:
        """Identify the best (most interpretable and monosemantic) latents in `activations`

        Parameters
        ----------
        labels : Sequence
            Labels for each sample in `activations`

        Returns
        -------
        Dataframe with the following columns:

        latent_idx: latent index
        label_max: label that the latent has the highest *average* activation for
        max_activation_avg: average of highest activation for `label_max` across samples
        max_activation: the highest activation observed for `label_max` across samples
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
        tmp: dict = {"latent_idx": self.lidx}
        ones, _, unique_labels = encode_labels(labels)
        ones = torch.tensor(ones).to(self.acts_dtype)
        avg_acts: Tensor = torch.matmul(self.acts.transpose(0, 1), ones) / ones.sum(
            dim=0
        )
        maxes, _idx = avg_acts.max(dim=1)
        idx = avg_acts.argmax(dim=1)
        tmp["max_activation_avg"] = maxes
        tmp["max_activation_prop"] = maxes / avg_acts.sum(axis=1)
        tmp["label_max"] = unique_labels[idx]

        # t(activations) has shape d_sae x n
        # ones  has shape n x k
        # expands to d_sae x n x 1
        # and       1 x n x k
        tmp["max_activation"], _ = max_by_label(self.acts, labels).max(dim=1)

        # Classifier scoring
        active_counts = torch.matmul(
            (self.acts > self.threshold).to(self.acts_dtype).transpose(0, 1), ones
        )
        inactive_counts = torch.matmul(
            (self.acts <= self.threshold).to(self.acts_dtype).transpose(0, 1), ones
        )

        tp = torch.take_along_dim(active_counts, idx.reshape(-1, 1), dim=1).flatten()
        fn = torch.take_along_dim(inactive_counts, idx.reshape(-1, 1), dim=1).flatten()
        tn = inactive_counts.sum(axis=1) - fn
        fp = active_counts.sum(axis=1) - tp
        tmp["f1"] = (2 * tp) / (2 * tp + fp + fn)
        tmp["precision"] = tp / (tp + fp)
        tmp["fpr"] = fp / (fp + tn)
        tmp["fnr"] = fn / (fn + tp)
        tmp["sensitivity"] = tp / (tp + fn)
        tmp["specificity"] = tn / (tn + fp)

        schema = {
            k: pl.Float64 for k in tmp.keys() if k not in {"label_max", "latent_idx"}
        }
        schema["latent_idx"] = pl.String
        schema["label_max"] = pl.Categorical
        return pl.DataFrame(tmp, schema=schema)


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
