#!/usr/bin/env ipython


from collections.abc import Sequence
from functools import reduce

import lightning as L
import numpy as np
import pandas as pd
import sklearn.preprocessing as sp
import torch
import torch.nn as nn
import torchmetrics.functional.classification as tmet
from amr_predict.utils import iter_cols
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


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


def multitask_metrics2df(metrics: dict) -> pd.DataFrame:
    to_df = {"task": [], "metric": [], "value": []}
    for task, dct in metrics.items():
        for metric, value in dct.items():
            if metric != "cm":
                to_df["task"].append(task)
                to_df["metric"].append(metric)
                to_df["value"].append(value.item())
    return pd.DataFrame(to_df)


def multitask_all_metrics(
    scores: Sequence[Tensor],
    y_true: Tensor,
    n_classes: Sequence[int],
    task_names: Sequence[str] | None = None,
) -> dict:
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
