#!/usr/bin/env ipython

from collections.abc import Sequence
from functools import reduce

import anndata as ad
import lightning as L
import numpy as np
import pandas as pd
import polars as pl
import sklearn.preprocessing as sp
import torch
import torch.nn as nn
import torchmetrics.functional.classification as tmet
from amr_predict.utils import iter_cols, vecdist
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.regression.mse import mean_squared_error
from torchmetrics.functional.regression.nrmse import normalized_root_mean_squared_error
from torchmetrics.functional.regression.pearson import pearson_corrcoef
from torchmetrics.functional.regression.spearman import spearman_corrcoef


def multitask_acc(
    predictions: Tensor | np.ndarray,
    y_true: Tensor | DataLoader | Dataset | np.ndarray,
    n_classes: Sequence[int],
    task_names: Sequence[str] | None = None,
    as_df: bool = False,
) -> dict | pd.DataFrame:
    """Compute accuracy independently on each prediction task

    Parameters
    ----------
    predictions : multitask predictions, same shape as y_true
    y_true : true values, of shape n_samples x n_tasks
    n_classes : iterable where the ith index is the number of classes in the ith task
    task_names : names of prediction tasks

    Returns
    -------
    Dictionary of task_name->task_accuracy. If names not provided, indices in
        y_true are used instead
    """
    if isinstance(y_true, Dataset):
        y_true = y_true[:][1]
    elif isinstance(y_true, DataLoader):
        y_true = y_true.dataset[:][1]
    elif isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(predictions, np.ndarray):
        predictions = torch.tensor(predictions)
    y_iter = iter_cols(y_true)
    pred_iter = iter_cols(predictions)
    if task_names is None:
        task_names = [str(i) for i in range(predictions.shape[1])]
    result = {}
    for i, (task, y, pred) in enumerate(zip(task_names, y_iter, pred_iter)):
        result[task] = tmet.accuracy(
            preds=pred, target=y, num_classes=n_classes[i], task="multiclass"
        ).item()
    if not as_df:
        return result
    df = {"metric": [], "value": [], "task": []}
    for task, val in result.items():
        df["metric"].append("acc")
        df["value"].append(val)
        df["task"].append(task)
    return pd.DataFrame(df)


def multitask_metrics2df(metrics: dict) -> pl.DataFrame:
    to_df = {"task": [], "metric": [], "value": []}
    for task, dct in metrics.items():
        for metric, value in dct.items():
            if metric != "cm":
                to_df["task"].append(task)
                to_df["metric"].append(metric)
                to_df["value"].append(value.item())
    return pl.DataFrame(to_df)


def multitask_all_reg(
    pred: Tensor, y_true: Tensor, task_names: Sequence[str] | None = None
) -> dict:
    result = {}
    if task_names is None:
        task_names = [str(i) for i in range(pred.shape[1])]
    for p, truth, task in zip(iter_cols(pred), iter_cols(y_true), task_names):
        result[task] = {}
        result[task]["mse"] = mean_squared_error(preds=p, target=truth)
        result[task]["spearman"] = spearman_corrcoef(preds=p, target=truth)
        result[task]["pearson"] = pearson_corrcoef(preds=p, target=truth)
        result[task]["nrmse"] = normalized_root_mean_squared_error(
            preds=p, target=truth
        )
    return result


def multitask_all_cls(
    scores: Sequence[Tensor],
    y_true: Tensor,
    n_classes: Sequence[int],
    task_names: Sequence[str] | None = None,
) -> dict:
    """Compute various multitask metrics for classification"""
    if y_true.shape[1] != len(scores):
        raise ValueError(
            "The given truth matrix does not match the sequence of scores!"
        )
    to_iter = iter_cols(y_true)
    if task_names is None:
        task_names = [str(i) for i in range(len(scores))]
    result = {}
    for task, truth, score, n in zip(task_names, to_iter, scores, n_classes):
        result[task] = {}
        result[task]["acc"] = tmet.accuracy(
            preds=score, target=truth, num_classes=n, task="multiclass"
        )  # NOTE: the multiclass_accuracy version produced a different result
        result[task]["kappa"] = tmet.cohen_kappa(
            preds=score, target=truth, num_classes=n, task="multiclass"
        )
        result[task]["mcc"] = tmet.matthews_corrcoef(
            preds=score, target=truth, num_classes=n, task="multiclass"
        )
        result[task]["auroc"] = tmet.auroc(
            preds=score, target=truth, num_classes=n, task="multiclass"
        )
        result[task]["aupr"] = tmet.average_precision(
            preds=score, target=truth, num_classes=n, task="multiclass"
        )
        result[task]["cm"] = tmet.confusion_matrix(
            preds=score, target=truth, num_classes=n, task="multiclass"
        )
    return result


def multitask_cross_entropy_loss(
    y_pred: Tensor,
    y_true: Tensor,
    weights: Tensor | None = None,
    model: L.LightningModule | None = None,
    prefix: str = "",
) -> Tensor:
    losses: Tensor = torch.empty(y_true.shape[1])
    for i, (task_pred, task_y) in enumerate(
        zip(y_pred, torch.unbind(y_true, dim=1))
    ):  # Gives y_hat = softmax(Xw + b)
        # tensor of shape n_samples, n_classes
        loss = nn.functional.cross_entropy(task_pred, task_y)
        if model is not None:
            name = f"loss_{i}" if not prefix else f"{prefix}_loss_{i}"
            model.log(name, loss)
        losses[i] = loss
        # Get loss on tasks separately
    if weights is not None and len(weights) == len(losses):
        losses = losses * weights
    return losses.sum()


class ConfusionMatrices:
    "Class for performing operations on a collection of confusion matrices"

    def __init__(
        self,
        matrices: list[pd.DataFrame | np.ndarray | Tensor],
        encoder: sp.LabelEncoder | None = None,
    ) -> None:
        init_shape = next(iter(matrices)).shape
        self.n_classes: int = init_shape[0]
        self.encoder: sp.LabelEncoder | None = encoder
        self.matrices: list[pd.DataFrame] = []
        for i, m in enumerate(matrices):
            self._add_cm(m, i)

    def _add_cm(self, m: pd.DataFrame | np.ndarray | Tensor, i):
        if m.shape[0] != m.shape[1]:
            raise ValueError(f"the {i}th confusion matrix is not square!")
        elif m.shape[0] != self.n_classes:
            raise ValueError(
                f"the {i} confusion matrix does not match the shape of the other matrices!"
            )
        if isinstance(m, pd.DataFrame):
            self.matrices.append(m)
        else:
            self.matrices.append(ConfusionMatrices.as_df(m, self.encoder))

    def add_cms(self, matrices: Sequence | pd.DataFrame):
        if not isinstance(matrices, Sequence):
            self._add_cm(matrices, 0)
        else:
            _ = [self._add_cm(m, i) for m, i in enumerate(matrices)]

    def total_correctness(self) -> pd.DataFrame:
        dfs = pd.concat(
            [
                ConfusionMatrices.correctness(m).loc[
                    :, ["label", "true_positives", "total_count"]
                ]
                for m in self.matrices
            ]
        )
        agg = dfs.groupby("label").agg("sum").reset_index()
        agg["accuracy"] = agg["true_positives"] / agg["total_count"]
        agg["label_prop"] = agg["total_count"] / agg["total_count"].sum()
        return agg

    def std_correctness(self) -> pd.DataFrame:
        dfs = pd.concat(
            [
                ConfusionMatrices.correctness(m).loc[:, ["label", "true_positives"]]
                for m in self.matrices
            ]
        )
        agg = dfs.groupby("label").agg("std").reset_index()
        return agg

    def mean(self) -> pd.DataFrame:
        """Return a single confusion matrix computed by averaging over all matrices"""
        return reduce(lambda x, y: x + y, self.matrices)

    @staticmethod
    def as_df(
        cm: Tensor | np.ndarray, encoder: sp.LabelEncoder | None = None
    ) -> pd.DataFrame:
        n_classes = cm.shape[0]
        labels = [i for i in range(n_classes)]
        labels = encoder.inverse_transform(labels) if encoder is not None else labels
        if isinstance(cm, Tensor):
            cm = cm.cpu().numpy()
        return pd.DataFrame(cm, columns=labels, index=labels)

    @staticmethod
    def correctness(cm: pd.DataFrame) -> pd.DataFrame:
        """Report the count of correct predictions for individual
        labels in confusion matrix `cm`, as well as accuracy
        Columns in `cm` are taken to be predictions, rows are truth
        """
        if cm.shape[0] != cm.shape[1]:
            raise ValueError("Given confusion matrix is not square!")
        total_counts = cm.sum(axis=1)
        tp = np.diag(cm)
        result = pd.DataFrame(
            {
                "label": list(cm.index),
                "true_positives": tp,
                "accuracy": tp / total_counts,
                "total_count": total_counts,
                "label_prop": total_counts / total_counts.sum(),
            }
        )
        return pd.DataFrame(result).reset_index(drop=True)


def format_cms(metric_dcts: list[dict], encoder: sp.LabelEncoder | None = None):
    """Format the confusion matrix results from a list of `multiclass_all_metrics`"""
    tasks = next(iter(metric_dcts)).keys()
    cms = []
    for dct in metric_dcts:
        for task in tasks:
            cm = dct[task]["cm"]
            cms.append(cm)
            cm_df: pd.DataFrame = ConfusionMatrices.as_df(cm, encoder)
            # cm_metrics =
    return {"label_metrics": [], "average": []}


def gini_impurity(x: np.ndarray) -> float:
    _, counts = np.unique(x, return_counts=True)
    rel_freq = counts / counts.sum()
    return np.sum(rel_freq * (1 - rel_freq))


def nn_proportions(
    adata: ad.AnnData,
    columns: Sequence,
    include_null: bool = True,
    include_impurity: bool = True,
    null_bootstrap_rounds: int = 1000,
    rng: int | None = None,
    **kws,
) -> dict[str, pl.DataFrame | int]:
    """For each sample, compute the proportion of nearest neighbors in the dataset that
    have the same values in `columns`

    Parameters
    ----------
    columns : Sequence
        columns of `adata.obs` to calculate proportion on
    kws : kwargs
        keyword arguments passed to sklearn.neighbors.NearestNeighbors

    Returns
    -------
    dict
        nn_prop: dataframe is of shape (n_samples, len(columns)). Each column is the
        proportion of the sample's nearest neighbors that were the same under that
        column name

        nn_dist: df of per-sample summary statistics of neighbor distances

        n_neighbors: number of neighbors used in calculation

        If `include_null`, the following are also included
            null_prop: dictionary denoting, for each column, the expected proportion of
                observations having the same values as the column for a randomly drawn
                sample of size n_neighbors, with respect to a single observation
            null_dist: the equivalent of null_prop, but for distance -
                the average distance between any random pair of observations in the dataset

        If `include_impurity`, also includes two keys for the Gini impurity (Gini index)
            of an NN cluster as well as the null equivalent
    """

    nclass = NearestNeighbors(**kws)
    nclass.fit(adata.X)
    distances, neighbors = nclass.kneighbors()
    df = adata.obs
    n_neighbors = neighbors.shape[1]
    tmp = {}
    impurity_tmp = {}
    for col in columns:
        var_mat = np.hstack(
            [
                df[col].values.reshape(-1, 1),
                np.vstack([df[col].iloc[n] for n in neighbors]),
            ]
        )
        tmp[col] = np.apply_along_axis(
            lambda x: (x[0] == x[1:]).sum() / n_neighbors, axis=1, arr=var_mat
        )
        if include_impurity:
            impurity_tmp[col] = np.apply_along_axis(gini_impurity, axis=1, arr=var_mat)
    dist_df = pl.DataFrame(distances)
    hcols = ("max", "mean", "min")
    exprs = [
        fn(pl.all()).alias(name)
        for name, fn in zip(
            hcols,
            (pl.max_horizontal, pl.mean_horizontal, pl.min_horizontal),
        )
    ]
    from_horizontal = dist_df.with_columns(*exprs).select(hcols)
    transposed = dist_df.transpose()
    from_transposed = pl.concat(
        [
            fn(transposed).transpose().rename({"column_0": name})
            for fn, name in zip(
                [pl.DataFrame.std, pl.DataFrame.var, pl.DataFrame.median],
                ["std", "var", "median"],
            )
        ],
        how="horizontal",
    )
    index_df = pl.DataFrame(adata.obs.index.values, schema=["index"])
    result = {
        "nn_prop": pl.concat([index_df, pl.DataFrame(tmp)], how="horizontal"),
        "nn_dist": pl.concat(
            [index_df, pl.concat([from_horizontal, from_transposed], how="horizontal")],
            how="horizontal",
        ),
        "n_neighbors": n_neighbors,
    }
    if include_impurity:
        result["nn_impurity"] = pl.concat(
            [index_df, pl.DataFrame(impurity_tmp)], how="horizontal"
        )
    if include_null:
        gen = np.random.default_rng(rng)
        rand_pairs = gen.choice(
            list(range(adata.shape[0])), (null_bootstrap_rounds, 2), replace=True
        )
        col_df: pd.DataFrame = adata.obs.loc[:, columns]
        nulls = {}
        nulls["null_dist"] = vecdist(
            adata.X[rand_pairs[:, 0], :],
            adata.X[rand_pairs[:, 1], :],
            metric=kws.get("metric", "cosine"),
        ).mean()
        for n in ["prop", "impurity"]:
            nulls[f"null_{n}"] = {c: [] for c in columns}
        for _ in range(null_bootstrap_rounds):
            cur = col_df.sample(n=n_neighbors, random_state=rng)
            for col in columns:
                nulls["null_prop"][col].append((cur[col] == cur[col].iloc[0]).sum())
                if include_impurity:
                    nulls["null_impurity"][col].append(gini_impurity(cur[col]))
        nulls["null_prop"] = (pd.DataFrame(nulls["null_prop"]) / n_neighbors).mean()
        nulls["null_impurity"] = pd.DataFrame(nulls["null_impurity"]).mean()
        result.update(nulls)
    return result
