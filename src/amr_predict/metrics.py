#!/usr/bin/env ipython

import itertools
from collections import Counter
from collections.abc import Callable, Sequence
from functools import reduce
from pathlib import Path
from typing import Any, Literal

import amr_predict.enums as ae
import anndata as ad
import jaxtyping
import lightning as L
import numpy as np
import pandas as pd
import polars as pl
import skbio.sequence as sks
import skbio.sequence.distance as ssd
import sklearn.model_selection as ms
import sklearn.preprocessing as sp
import torch
import torch.nn as nn
import torchmetrics.functional.classification as tmet
from amr_predict.cache import LinkedDataset
from amr_predict.taxonomy import TaxonomyTree
from amr_predict.utils import expand_annotations, iter_cols, resample_pairs, vecdist
from attrs import Factory, define, field, validators
from beartype import beartype
from loguru import logger
from numpy.random import Generator
from scipy import stats
from scipy.spatial.distance import pdist
from scipy.stats import ecdf
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.metrics.pairwise import paired_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.regression.mse import mean_squared_error
from torchmetrics.functional.regression.nrmse import normalized_root_mean_squared_error
from torchmetrics.functional.regression.pearson import pearson_corrcoef
from torchmetrics.functional.regression.spearman import spearman_corrcoef

logger.disable("amr_predict")


SEQ_DISTANCE_METRICS: dict[str, Callable] = {
    # "pdist": ssd.pdist,
    # "paralin": ssd.paralin,
    "kmer": ssd.kmer_distance,
    # "logdet": ssd.logdet,
}


def random_neighbor_score(
    natural: LinkedDataset,
    random: LinkedDataset,
    n_neighbors: int = 10,
    prop: float = 0.8,
    iterations: int = 10,
    seed: int | None = None,
    nn_kws: dict | None = None,
) -> float:
    def _run_once() -> float:
        n_subsampled = natural.sample(fraction=prop, seed=seed)
        r_subsampled = random.sample(fraction=prop, seed=seed)
        # 1 for random, 0 for natural
        labels = np.array([0] * len(n_subsampled) + [1] * len(r_subsampled))
        nn: NearestNeighbors = NearestNeighbors(
            **(nn_kws or {"n_neighbors": n_neighbors})
        )
        x = [
            n_subsampled[n_subsampled.x_key].numpy(),
            r_subsampled[r_subsampled.x_key].numpy(),
        ]
        nn.fit(np.vstack(x))
        _, neighbors = nn.kneighbors(n_subsampled[n_subsampled.x_key].numpy())
        dist = np.array([(labels[row] == 1).sum() / n_neighbors for row in neighbors])
        return dist.mean()

    return np.array([_run_once() for _ in range(iterations)]).mean()


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
    pred: Tensor,
    y_true: Tensor,
    task_names: Sequence[str] | None = None,
    metrics: tuple = ("mse", "pearson", "spearman", "nrmse"),
) -> dict:
    result = {}
    if task_names is None:
        task_names = [str(i) for i in range(pred.shape[1])]
    for p, truth, task in zip(iter_cols(pred), iter_cols(y_true), task_names):
        result[task] = {}
        for m in metrics:
            if m == "mse":
                score = mean_squared_error(preds=p, target=truth)
            elif m == "spearman":
                score = spearman_corrcoef(preds=p, target=truth)
            elif m == "pearson":
                score = pearson_corrcoef(preds=p, target=truth)
            elif m == "nrmse":
                score = normalized_root_mean_squared_error(preds=p, target=truth)
            else:
                raise ValueError(f"Metric {m} not supported")
            result[task][m] = score
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
    y_pred: tuple[Tensor],
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


def gini_impurity(x: np.ndarray | Tensor) -> float:
    if isinstance(x, np.ndarray):
        _, counts = np.unique(x, return_counts=True)
        rel_freq = counts / counts.sum()
        return np.sum(rel_freq * (1 - rel_freq))
    _, counts = torch.unique(x, return_counts=True)
    rel_freq = counts / counts.sum()
    return torch.sum(rel_freq * (1 - rel_freq)).item()


@define
class McTestResult:
    observed_dist: jaxtyping.Shaped[Any, "a"]
    rvs_dist: jaxtyping.Shaped[Any, "a"]
    p_values: jaxtyping.Shaped[Any, "a"]
    alternative: str
    p_adj: float

    mean_null: float = field(default=None, init=False)
    std_null: float = field(default=None, init=False)
    median_null: float = field(default=None, init=False)
    iqr_null: float = field(default=None, init=False)

    mean: float = field(init=False, default=None)
    std: float = field(init=False, default=None)
    median: float = field(init=False, default=None)
    iqr: float = field(init=False, default=None)

    def __attrs_post_init__(self):
        for use_null in (True, False):
            for field in ("mean", "std", "median", "iqr"):
                name = f"{field}_null" if use_null else field
                dist = self.rvs_dist if use_null else self.observed_dist
                if field == "mean":
                    val = dist.mean()
                elif field == "std":
                    val = dist.std()
                elif field == "iqr":
                    val = stats.iqr(dist)
                elif field == "median":
                    val = (
                        dist.median()
                        if isinstance(dist, torch.Tensor)
                        else np.median(dist)
                    )
                setattr(self, name, val)

    def to_pl(self) -> pl.DataFrame:
        stat_fields = ("mean", "std", "median", "iqr")
        wanted_fields = (
            ("p_adj", "alternative")
            + stat_fields
            + tuple([f"{f}_null" for f in stat_fields])
        )
        return pl.DataFrame({f: getattr(self, f) for f in wanted_fields})


def mc_multinomial_test(
    observations: Sequence,
    n: int,
    all_categories: Sequence,
    multiple_test: bool = False,
    n_resample: int = 10_000,
    seed: None | int | Generator = None,
    log: bool = True,
) -> McTestResult:
    """Multinomial Monte Carlo test

    Parameters
    ----------
    n_resample : int
        Number of random draws used for to make the MC null

    Returns
    -------
    Tuple of p-values and adjusted combined p-value (Bonferroni correction)
    if a multiple test

    Notes
    -----
    The test statistic is the probability of an observation under
    the multinomial distribution

    This test can be used in place of Chi square when observed counts are
    small e.g. to if the label distribution of a
    very small subsample from a dataset follows the
    label distribution in the entire dataset
    """
    # TODO: make a note of this approach in `hypothesis testing`
    freq = pl.Series(all_categories).value_counts(normalize=True).sort("")
    names = freq[""].to_list()
    mnl = stats.multinomial(n=n, p=freq["proportion"], seed=seed)

    def sort_obs(obs: Sequence) -> np.ndarray:
        counted = Counter(obs)
        return np.array([counted.get(n, 0) for n in names])

    if multiple_test:
        ordered = np.vstack([sort_obs(o) for o in observations])
    else:
        ordered = np.array([sort_obs(observations)])

    observed_pr = mnl.logpmf(ordered) if log else mnl.pmf(ordered)
    randoms = mnl.rvs(size=n_resample)
    null_dist = mnl.logpmf(randoms) if log else mnl.pmf(ordered)
    ecdf = stats.ecdf(null_dist)
    p_values = ecdf.cdf.evaluate(observed_pr)

    return McTestResult(
        p_values=p_values,
        observed_dist=observed_pr,
        rvs_dist=null_dist,
        alternative="less",
        p_adj=min(len(p_values) * min(p_values), 1),
    )


@beartype
def generalized_mc_test(
    data: Tensor,
    statistic: Callable[[Tensor], Tensor],
    rvs: Callable[[int | None], Tensor],
    alternative: Literal["less", "greater", "two-sided"] = "two-sided",
    vectorized: bool = False,
    n_resample: int | None = None,
    obs_dim: int = 0,
) -> McTestResult:
    """Perform a generalized Monte Carlo test for test statistics on single observations

    The resulting p-values are combined
    by the ... method, which assumes no dependency structure
    """
    if vectorized:
        null_distribution = statistic(rvs()).numpy()
        observed = statistic(data).numpy()
    elif n_resample is None:
        raise ValueError("Number of resamplings must be provided if not vectorized")
    else:
        null_distribution = np.array(
            [statistic(rv) for rv in torch.unbind(rvs(n_resample), dim=obs_dim)]
        )
        observed = np.array([statistic(d) for d in torch.unbind(data, dim=obs_dim)])
    dist_ecdf = ecdf(null_distribution)
    proba = dist_ecdf.cdf.evaluate(observed)
    # Right-tailed test for proportion i.e. probability of observing
    #   proportion p > `observed_prop` under the null, as the higher the proportion the better
    if alternative == "less":  # percentage of null leq observed test
        # statistic
        p_value = proba
    elif alternative == "greater":  # percentage of null geq observed
        # test statistic
        p_value = 1 - proba
    elif alternative == "two-sided":
        p_value = np.min(np.vstack([proba, 1 - proba]), axis=0) * 2
    p_value[p_value == 0] = 1e-10
    # Combine using Bonferroni method
    # TODO: can use something more powerful, see the function "harmonic_mean_pvalue" in
    # test_metrics
    p_adj = min(len(p_value) * min(p_value), 1)
    return McTestResult(
        p_values=p_value,
        p_adj=p_adj,
        alternative=alternative,
        observed_dist=observed,
        rvs_dist=null_distribution,
    )


def nn_proportions(
    adata: ad.AnnData,
    columns: Sequence,
    include_null: bool = True,
    include_impurity: bool = True,
    null_bootstrap_rounds: int = 1000,
    rng: int | None = None,
    prefit: NearestNeighbors | None = None,
    **kws,
) -> dict[str, pl.DataFrame | int]:
    """For each sample, compute the proportion of nearest neighbors in the dataset that
    have the same values in `columns`

    Parameters
    ----------
    columns : Sequence
        columns of `adata.obs` to calculate proportion on
    nn_obj : NearestNeighbors
        pre-fit nearest neighbors object
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
    nn_obj: NearestNeighbors = prefit or NearestNeighbors(**kws)
    if prefit is None:
        nn_obj.fit(adata.X)
    logger.info("Kneighbor computation started")
    distances, neighbors = nn_obj.kneighbors()
    logger.success("Kneighbors fitted")
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
        # Simple permutation testing
        logger.info("Computing p-value with permutation test")
        gen = np.random.default_rng(rng)
        rand_pairs = gen.choice(
            list(range(adata.shape[0])), (null_bootstrap_rounds, 2), replace=True
        )
        logger.info(f"Shape of rand_pairs: {rand_pairs.shape}")
        col_df: pd.DataFrame = adata.obs.loc[:, columns]
        nulls, ecdfs = {}, {}
        observed_dist = result["nn_dist"]["mean"].mean()
        null_dist = paired_distances(
            adata.X[rand_pairs[:, 0], :],
            adata.X[rand_pairs[:, 1], :],
            metric=kws.get("metric", "cosine"),
        )
        dist_ecdf = ecdf(null_dist)
        nulls["null_dist"] = pl.DataFrame(
            {
                "p_value": [float(dist_ecdf.cdf.evaluate(observed_dist))],
                "observed_avg": [observed_dist],
                "null_avg": [null_dist.mean()],
            }
        )
        for n in ["prop", "impurity"]:
            nulls[f"null_{n}"] = {c: [] for c in columns}
            ecdfs[f"null_{n}"] = {
                "column": [],
                "p_value": [],
                "observed_avg": [],
            }
        for _ in range(null_bootstrap_rounds):
            cur = col_df.sample(n=n_neighbors, random_state=rng)
            for col in columns:
                nulls["null_prop"][col].append((cur[col] == cur[col].iloc[0]).sum())
                if include_impurity:
                    nulls["null_impurity"][col].append(gini_impurity(cur[col]))
        for col in columns:
            for val in ("prop", "impurity"):
                if not include_impurity and val == "impurity":
                    continue
                prop_ecdf = ecdf(nulls[f"null_{val}"][col])
                observed_avg = result[f"nn_{val}"][col].mean()
                proba = prop_ecdf.cdf.evaluate(observed_avg)
                proba = 1 - proba if val == "prop" else proba
                # Right-tailed test for proportion i.e. probability of observing
                #   proportion p > `observed_prop` under the null, as the higher the proportion the better
                # Left-tailed for impurity
                ecdfs[f"null_{val}"]["column"].append(col)
                ecdfs[f"null_{val}"]["p_value"].append(float(proba))
                ecdfs[f"null_{val}"]["observed_avg"].append(observed_avg)
        nulls["null_prop"] = (
            (pl.DataFrame(nulls["null_prop"]) / n_neighbors)
            .mean()
            .unpivot(value_name="null_avg", variable_name="column")
            .join(pl.DataFrame(ecdfs["null_prop"]), on="column")
        )
        if include_impurity:
            nulls["null_impurity"] = (
                pl.DataFrame(nulls["null_impurity"])
                .mean()
                .unpivot(value_name="null_avg", variable_name="column")
                .join(pl.DataFrame(ecdfs["null_impurity"]), on="column")
            )
        result.update(nulls)
    return result


# * Embedding correlation


@define
class EmbeddingCorrelations:
    dset: LinkedDataset
    columns: dict[str, ae.SeqCovariates]
    embedding_distance: Literal["cosine", "euclidean", "manhattan"]
    sequence_distance: str = field(
        default="kmer", validator=validators.in_(SEQ_DISTANCE_METRICS.keys())
    )
    anno_sep: str = ";"
    seed: int | None = None
    n_resample: int = 10_000
    tree_file: Path | str | None = None
    n: int = field(
        init=False, default=Factory(lambda self: self.dset.shape[0], takes_self=True)
    )
    tree: TaxonomyTree | None = field(
        init=False,
        default=Factory(
            lambda self: TaxonomyTree(self.tree_file)
            if (
                self.tree_file
                and ae.SeqCovariates.taxonomic_similarity in self.columns.values()
            )
            else None,
            takes_self=True,
        ),
    )
    seq_kws: dict[str, dict] = field(
        factory=lambda: {"kmer": {"k": 5}},
        validator=validators.deep_mapping(
            key_validator=validators.in_(SEQ_DISTANCE_METRICS)
        ),
    )

    def _dist_sequence(self, col: str, idx1, idx2) -> np.ndarray:
        first, sec = self.dset[col][idx1], self.dset[col][idx2]
        fn = SEQ_DISTANCE_METRICS[self.sequence_distance]
        kws = self.seq_kws[self.sequence_distance]
        result = [
            sks.Sequence(f).distance(
                sks.Sequence(s), metric=lambda x, y: fn(x, y, **kws)
            )
            for f, s in zip(first, sec)
        ]
        return np.array(result)

    def _dist_taxonomic(self, col: str, idx1, idx2) -> np.ndarray:
        first, sec = self.dset[col][idx1], self.dset[col][idx2]
        if self.tree is None:
            raise ValueError(
                "Must provide NCBI taxonomy dump file to use taxonomic distance"
            )
        return np.array([self.tree.dist(f, s) for f, s in zip(first, sec)])

    # TODO: Can also include a simplified measure of structural similarity
    # for proteins e.g. biochemical properties, predictions with PyMOL??

    def _dist_functional(self, col: str, idx1, idx2) -> np.ndarray:
        expanded = expand_annotations(self.dset[col], split=self.anno_sep)
        first, sec = expanded[idx1, :], expanded[idx2, :]

        # Elementwise jaccard distance (vectorized)
        numer = np.logical_xor(first, sec).sum(axis=1)
        denom = numer + (first & sec).sum(axis=1)
        return numer / denom

    def _run(self) -> tuple[pl.DataFrame, dict]:
        """
        Compute correlation in batches
        """
        pairs: np.ndarray = resample_pairs(
            x=list(range(self.n)), n=self.n_resample, rng=self.seed
        )
        p1, p2 = pairs[:, 0], pairs[:, 1]
        emb_dist = vecdist(
            self.dset[p1]["x"].numpy(),
            self.dset[p2]["x"].numpy(),
            metric=self.embedding_distance,
        )
        dists = {"embedding": emb_dist}  # For visualization
        correlations: dict = {}
        for col, covar in self.columns.items():
            if covar == ae.SeqCovariates.functional_similarity:
                cov_dist = self._dist_functional(col, p1, p2)
            elif covar == ae.SeqCovariates.sequence_similarity:
                cov_dist = self._dist_sequence(col, p1, p2)
            elif covar == ae.SeqCovariates.taxonomic_similarity:
                cov_dist = self._dist_taxonomic(col, p1, p2)
            else:
                raise NotImplementedError()
            dists[col] = cov_dist
            correlations[col] = stats.spearmanr(emb_dist, cov_dist)
        return pl.DataFrame(correlations).with_columns(
            pl.Series(["statistic", "p_value"]).alias("metric")
        ), dists

    def run(self, rounds: int = 1) -> tuple[pl.DataFrame, dict]:
        if rounds == 1:
            return self._run()
        dfs, dists = [], {}
        for i in range(rounds):
            df, dist = self._run()
            dfs.append(df.with_columns(pl.lit(i + 1).alias("round")))
            if not dists:
                dists.update(dist)
            else:
                for k, v in dist.items():
                    dists[k] = np.concat(dists[k], v)
        return pl.concat(dfs), dists


# * Nearest neighbors


@define
class NeighborMetrics:
    """Helper class for running analyses on embeddings in NN space

    Parameters
    ----------
    anno_sep : str
        Separator delimiting individual annotations
    subsample : float | None
        Proportion of samples from the dataset to use as reference points for metric
        calculations
        If set to None, don't subsample and use every sample as a reference point

    n_resample : int
        - Number of resamples for permutation test
        -
    """

    dset: LinkedDataset
    category_cols: Sequence[str]
    anno_cols: Sequence[str]
    anno_sep: str = ";"
    subsample: float | None = field(
        default=None,
        validator=validators.or_(
            validators.in_([None]), validators.and_(validators.gt(0), validators.lt(1))
        ),
    )
    seed: int | None = None
    rng: Generator = field(
        init=False,
        default=Factory(lambda self: np.random.default_rng(self.seed), takes_self=True),
    )
    n_neighbors: int = 8
    nn_kws: dict = field(factory=dict)
    n_resample: int = 10_000
    sampled_idx: np.ndarray = field(init=False, default=None)
    nn: NearestNeighbors = field(
        default=Factory(
            lambda self: NearestNeighbors(
                **(self.nn_kws | {"n_neighbors": self.n_neighbors})
            ),
            takes_self=True,
        )
    )
    n: int = field(
        init=False, default=Factory(lambda self: self.dset.shape[0], takes_self=True)
    )
    encoders: dict[str, LabelEncoder] = field(init=False, factory=dict)

    def __attrs_post_init__(self):
        if self.subsample is None:
            self.sampled_idx = np.array(range(self.n))
        else:
            n = round(self.subsample * self.n)
            self.sampled_idx = self.rng.choice(range(self.n), size=n)
        logger.info("Kneighbor computation started")
        self.nn.fit(self.dset[self.dset.x_key][self.sampled_idx])
        logger.success("Kneighbors fitted")

    def _random_neighbors(self, x: np.ndarray, n: int | None = None) -> Tensor:
        n = n or self.n_resample
        indices = range(x.shape[0])
        call = lambda: self.rng.choice(indices, self.n_neighbors + 1, replace=False)
        if len(x.shape) == 1:
            rands = [torch.tensor(x[call()]) for _ in range(n)]
            return torch.vstack(rands)
        nn = torch.stack([torch.tensor(x[call(), :]) for _ in range(n)])
        assert len(nn.shape) == 3 and nn.shape[0] == n
        return nn

    def _neighbor_prop(self, x: np.ndarray) -> float:
        """
        Test statistic returning the difference between observed and random neighborhood
        annotation/category proportions

        The higher, the better
        """
        sum_axis = 1
        if len(x.shape) == 3:  # first axis becomes the permutations
            ref_points = x[:, :, 0].reshape(-1, 1)
            neighbors = x[:, :, 1:].reshape(-1, x.shape[-1] - 1)
        elif len(x.shape) == 2:
            ref_points = x[:, 0].reshape(-1, 1)
            neighbors = x[:, 1:]
        else:
            ref_points = x[0]
            neighbors = x[1:]
            sum_axis = -1
        return (ref_points == neighbors).sum(axis=sum_axis) / self.n_neighbors

    @beartype
    @staticmethod
    def _get_category_matrix(
        categories,
        neighbors: jaxtyping.Shaped[Any, "a b"],
        indices: jaxtyping.Shaped[Any, "a"],
        randomize: bool = False,
        seed: int | None = None,
    ) -> tuple[LabelEncoder, Tensor, Tensor]:
        encoder = LabelEncoder()
        cats = pl.Series(encoder.fit_transform(categories))
        neighbor_subset = neighbors[indices, :]
        if randomize:
            cats = cats.shuffle(seed=seed)
        category_mat: Tensor = torch.hstack(
            [
                torch.tensor(cats[indices]).reshape(-1, 1),
                torch.vstack([torch.tensor(cats[n]) for n in neighbor_subset]),
            ]
        )
        # Array of shape: (samples, 1 + n_neighbors)
        return encoder, category_mat, cats.to_torch()

    def _compute_category_metrics(
        self, indices, category_col: str, randomize: bool = False
    ) -> tuple[McTestResult, McTestResult, McTestResult]:
        _, neighbors = self.nn.kneighbors(self.dset[self.dset.x_key][indices])
        encoder, category_mat, cats = self._get_category_matrix(
            self.dset[category_col],
            neighbors,
            indices,
            randomize=randomize,
            seed=self.seed,
        )
        self.encoders[category_col] = encoder
        n_prop_result = generalized_mc_test(
            category_mat,
            statistic=self._neighbor_prop,
            rvs=lambda: self._random_neighbors(cats),
            vectorized=True,
            alternative="greater",
        )
        n_prop_result_mnl = mc_multinomial_test(
            category_mat,
            n=self.n_neighbors + 1,
            all_categories=cats,
            n_resample=self.n_resample,
            seed=self.seed,
            log=True,
        )
        impurity_result = generalized_mc_test(
            category_mat,
            statistic=gini_impurity,
            rvs=lambda x: self._random_neighbors(cats, x),
            n_resample=self.n_resample,
            alternative="less",  # Lower impurity the better
            vectorized=False,
        )
        return n_prop_result, n_prop_result_mnl, impurity_result

    def _compute_anno_metrics(
        self, indices, anno_col: str, randomize: bool = False
    ) -> McTestResult:
        binary_annots: np.ndarray = expand_annotations(
            self.dset[anno_col].shuffle(seed=self.seed)
            if randomize
            else self.dset[anno_col],
            self.anno_sep,
        )
        observed = binary_annots[indices, :]
        _, neighbors = self.nn.kneighbors(self.dset[self.dset.x_key][indices])
        # shape of n x n_neighbors
        anno_mat: np.ndarray = np.hstack(
            [
                observed.reshape(-1, 1, observed.shape[1]),
                np.vstack([[binary_annots[n_idx, :]] for n_idx in neighbors]),
            ]
        )
        res = generalized_mc_test(
            data=torch.from_numpy(anno_mat),
            rvs=lambda x: self._random_neighbors(binary_annots, x),
            n_resample=self.n_resample,
            statistic=lambda x: pdist(x, metric="jaccard").mean(),
            vectorized=False,
        )
        return res

    def run(
        self, with_randomization: bool = False
    ) -> tuple[pl.DataFrame, dict[str, jaxtyping.Shaped[Any, "a"]]]:
        dfs = []
        distributions = {
            "gini_impurity": {},
            "neighbor_proportion": {},
            "mean_jaccard_distance": {},
        }
        for col in itertools.chain(self.category_cols, self.anno_cols):
            for do_random in (False, True):
                if not with_randomization and do_random:
                    continue
                if col in self.category_cols:
                    nn, nn_mnl, gini = self._compute_category_metrics(
                        self.sampled_idx, category_col=col, randomize=do_random
                    )
                    df = pl.concat(
                        [gini.to_pl(), nn_mnl.to_pl(), nn.to_pl()],
                        how="diagonal_relaxed",
                    ).with_columns(
                        pl.Series(
                            [
                                "gini_impurity",
                                "neighbor_proportion_multinomial_test",
                                "neighbor_proportion",
                            ]
                        ).alias("metric"),
                        pl.lit(col).alias("column"),
                    )
                    distributions["gini_impurity"][col] = gini.observed_dist
                    distributions["neighbor_proportion"][col] = nn.observed_dist
                else:
                    res = self._compute_anno_metrics(
                        self.sampled_idx, anno_col=col, randomize=do_random
                    )

                    df = res.to_pl().with_columns(
                        metric=pl.lit("mean_jaccard_distance"), column=pl.lit(col)
                    )
                    distributions["mean_jaccard_distance"][col] = res.observed_dist
                if with_randomization:
                    df = df.with_columns(pl.lit(do_random).alias("randomized"))
                dfs.append(df)
        return pl.concat(dfs, how="diagonal_relaxed"), distributions


# * Perturbation metrics


@define
class PerturbationMetrics:
    id_col: str
    embedding_distance: Literal["cosine", "euclidean", "manhattan"]
    natural: LinkedDataset
    perturbed: LinkedDataset | None
    random: LinkedDataset | None
    level: Literal["seqs", "tokens"] = "seqs"
    classifier_kws: dict = field(factory=dict)
    seed: int | None = None
    random_is_pairable: bool = False
    rng: Generator = field(
        init=False,
        default=Factory(lambda self: np.random.default_rng(self.seed), takes_self=True),
    )
    classifier_name: str = "Lasso"
    n_distance_resampling: int = 5
    classifier: BaseEstimator = field(init=False)
    cv: int = 5
    rns_kws: dict = field(factory=lambda: {"prop": 1, "iterations": 20})
    subsample_prop: dict | None = field(
        factory=lambda: {"natural": 0.8, "perturbed": 0.8, "random": 0.8},
        validator=validators.or_(
            validators.in_((None,)),
            validators.deep_mapping(
                key_validator=validators.in_(
                    ["natural", "perturbed", "random"],
                ),
                value_validator=validators.instance_of(float),
            ),
        ),
    )
    idx_df: pl.DataFrame = field(init=False, default=None)

    @classifier.default
    def _classifier_dispatch(self) -> BaseEstimator:
        if self.classifier_name == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(**self.classifier_kws)
        elif self.classifier_name == "Lasso":
            from sklearn.linear_model import LogisticRegressionCV

            self.classifier_kws["l1_ratios"] = [1, 0]
            self.classifier_kws["solver"] = "saga"
            return LogisticRegressionCV(**self.classifier_kws)
        elif self.classifier_name == "SVM":
            from sklearn.svm import SVC

            return SVC(**self.classifier_kws)
        raise NotImplementedError()

    def _get_idx_df(self, dset) -> pl.DataFrame:
        df = dset.to_pl().select(self.id_col).with_row_index()
        if self.level == "tokens":
            df = df.unique(self.id_col)
        return df

    def __attrs_post_init__(self):
        if self.subsample_prop:
            for k, v in self.subsample_prop.items():
                dset: LinkedDataset = getattr(self, k)
                sampled = dset.sample(fraction=v, seed=self.seed)
                setattr(self, k, sampled)

        n_df = self._get_idx_df(self.natural).rename({"index": "natural"})
        p_df = self._get_idx_df(self.perturbed).rename({"index": "perturbed"})
        shared = n_df.join(p_df, on=self.id_col, how="inner")
        if self.random_is_pairable:
            r_df = self._get_idx_df(self.random).rename({"index": "random"})
            shared = shared.join(r_df, on=self.id_col, how="inner")
        else:
            shared = shared.with_columns(
                pl.Series(
                    self.rng.choice(range(len(self.random), size=shared.height))
                ).alias("random")
            )
        self.idx_df = shared

    def establish_baseline(
        self, split_kws: dict | None = None, n_repeats: int = 5
    ) -> pl.DataFrame:
        result = {"round": [], "mcc": []}
        indices = np.array(range(self.natural.shape))
        for i in range(n_repeats):
            labels = self.rng.choice([0, 1], size=self.natural.shape, replace=True)
            train_idx, test_idx = ms.train_test_split(indices, **(split_kws or {}))
            train = self.natural[train_idx][self.natural.x_key].numpy()
            test = self.natural[test_idx][self.natural.x_key].numpy()
            self.classifier.fit(train, y=labels[train])
            pred = self.classifier.predict(test)
            result["round"].append(i)
            result["mcc"].append(matthews_corrcoef(y_true=labels[test], y_pred=pred))
        return pl.DataFrame(result)

    def _combine_dsets_balanced(
        self, d1: LinkedDataset, d2: LinkedDataset
    ) -> tuple[np.ndarray, int]:
        d1_len, d2_len = len(d1), len(d2)
        if d1_len > d2_len:
            d1 = d1[self.rng.choice(range(d1_len, size=d2_len))]
            half_len = d2_len
        else:
            d2 = d1[self.rng.choice(range(d2_len, size=d1_len))]
            half_len = d1_len
        return np.vstack([d1[d1.x_key].numpy(), d2[d2.x_key].numpy()]), half_len

    def _distance_correlation(self) -> pl.DataFrame:
        """Compare the distributions of embedding distances under
        perturbation and randomization

        Notes
        -----
        When possible, the method pairs up sequences for
        distance computation.

        It uses a paired test instead because the calculation
        of each distance distribution
        uses the initial ids from the natural sequences
        """
        dists = {}
        result = {"distances_x": [], "distances_y": [], "statistic": [], "p_value": []}
        idx1 = self.idx_df["natural"]
        for pair_name, to_compare in zip(
            ["natural-random", "natural-perturbed", "natural-natural"],
            [self.random, self.perturbed, self.natural],
        ):
            _, compare_name = pair_name.split("-")
            idx2 = (
                self.idx_df[compare_name]
                if compare_name != "natural"
                else self.rng.choice(range(len(self.natural)), size=len(idx1))
            )
            dists[pair_name] = vecdist(
                self.natural[idx1][self.natural.x_key],
                to_compare[idx2][to_compare.x_key],
                metric=self.embedding_distance,
            )
        for x, y in [
            ("natural-random", "natural-natural"),
            ("natural-perturbed", "natural-natural"),
            ("natural-perturbed", "natural-random"),
        ]:
            test = stats.wilcoxon(x=dists[x], y=dists[y])
            result["distances_x"].append(x)
            result["distances_y"].append(y)
            result["statistic"].append(test.statistic)
            result["p_value"].append(test.pvalue)
        return pl.DataFrame(result)

    def _measure_classifiability(self):
        result = {"group": [], "mcc": []}
        for group, dset_pair in zip(
            ["cv_natural_w_perturbed", "cv_natural_w_random"],
            [(self.natural, self.perturbed), (self.natural, self.random)],
        ):
            x, half_len = self._combine_dsets_balanced(d1=dset_pair[0], d2=dset_pair[1])
            cv_results: dict = ms.cross_validate(
                self.classifier,
                X=x,
                y=[0] * half_len + [1] * half_len,
                scoring=make_scorer(matthews_corrcoef),
                cv=self.cv,
                estimator=group == "cv_natural_w_random",
                return_estimator=True,
            )
            scores = cv_results["test_score"]
            result["group"].extend([group] * len(scores))
            result["mcc"].extend(scores)
            if group == "cv_natural_w_random":
                x_test = self.perturbed[self.perturbed.x_key][:]
                len_test = len(x_test)
                for est in cv_results["estimators"]:
                    pred = est.predict(x_test)
                    result["group"].append("cv_natural_w_random_on_perturbed")
                    result["mcc"].append(
                        matthews_corrcoef(y_true=[0] * len_test, y_pred=pred)
                    )
        return pl.DataFrame(result)

    def _random_neighbor_score(self) -> dict:
        return {
            "natural": random_neighbor_score(
                natural=self.natural, random=self.random, **self.rns_kws
            ),
            "perturbed": random_neighbor_score(
                natural=self.perturbed, random=self.random, **self.rns_kws
            ),
        }

    def run(self) -> dict:
        return {
            "classifiability": self._measure_classifiability(),
            "random_neighbor_score": pl.DataFrame(self._random_neighbor_score()),
            "distance_correlation": self._distance_correlation(),
        }


# * Neighbor-preserving score
def seq_nn(x: np.ndarray, **kws) -> NearestNeighbors:
    """
    Compute nearest neighbors to sequence
    """

    # TODO: need to use something faster
    # Maybe https://github.com/nmslib/nmslib
    # See here for how to do custom
    # https://github.com/nmslib/nmslib/blob/master/manual/extend.md
    #
    # Another approach would be to map genomic distance to vectors in euclidean
    # space
    def seqdist(x, y, max_value):
        x_seqid, x_start, x_end = x
        y_seqid, y_start, y_end = y
        if x_seqid != y_seqid:
            return max_value
        return max(x_start - y_end, y_start - x_end, 0)

    nn = NearestNeighbors(
        **kws, metric=lambda x, y: seqdist(x, y, np.iinfo(np.int64).max)
    )
    encoder = LabelEncoder()
    if x[:, 0].dtype == "O":
        x[:, 0] = encoder.fit_transform(x[:, 0])
    nn.fit(x)
    return nn


def random_from_template(
    nrows: int, template: np.ndarray, rng: int | None | Generator = None
) -> np.ndarray:
    """Create a random 2d matrix, where the (min, max)
    values are defined per-dimension from `template`

    Parameters
    ----------
    template : np.ndarray
        Array of shape n_obs x n_vars.


    Returns
    -------
    A matrix of shape (nrows x n_vars), where the ith column is randomly sampled from
        a uniform distribution with min(template[:, i]) and max(template[:, i])
    """
    _, ncols = template.shape
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)
    mins = template.min(axis=0)
    maxes = template.max(axis=0)
    random_cols = [rng.uniform(mins[i], maxes[i], (nrows, 1)) for i in range(ncols)]
    return np.hstack(random_cols)


def nps(
    adata: ad.AnnData,
    seq_start_col: str,
    seq_end_col: str,
    seq_id_col: str,
    rng: int | None = None,
    prefit: NearestNeighbors | None = None,
    **kws,
):
    """Compute the neighbor-preserving score [1] for a set of sequence embeddings

    Parameters
    ----------
    param : argument
    prefit : NearestNeighbors
        A nn instance pre-fitted to adata.X

    Returns
    -------
    Neighbor-preserving score for the embeddings represented by adata.X
    """

    def len_intersections(arr: np.ndarray, x_idx: int = 0, y_idx: int = 1):
        return np.array(
            [
                len(
                    np.intersect1d(
                        arr[x_idx, i, :], arr[y_idx, i, :], assume_unique=False
                    )
                )
                for i in range(arr.shape[1])
            ]
        )

    nn_obj = prefit or NearestNeighbors(**kws)
    if prefit is None:
        nn_obj.fit(adata.X)
    kws = {k: v for k, v in kws.items() if k != "metric"}
    seq_nn_obj: NearestNeighbors = seq_nn(
        adata.obs.loc[:, [seq_id_col, seq_start_col, seq_end_col]].values, **kws
    )
    seq_neighbors = seq_nn_obj.kneighbors(return_distance=False)
    neighbors = np.array([nn_obj.kneighbors(return_distance=False), seq_neighbors])
    n_neighbors = neighbors.shape[2]
    len_intersect = len_intersections(neighbors)
    qnpr = (len_intersect / n_neighbors).mean()
    # rNPR is the NPR computed with random embeddings as queries, rather than observed embeddings
    random_embeds = random_from_template(neighbors.shape[1], adata.X, rng=rng)
    random_neighbors = np.array(
        [nn_obj.kneighbors(random_embeds, return_distance=False), seq_neighbors]
    )
    rnpr = (len_intersections(random_neighbors) / n_neighbors).mean()
    res = np.max(np.log10((qnpr + 1e-10) / (rnpr + 1e-10)), 0)
    return res
