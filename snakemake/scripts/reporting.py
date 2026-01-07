#!/usr/bin/env ipython

import os
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from itertools import batched
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import plotnine as gg
import polars as pl
import scanpy as sc
from amr_predict.metrics import nn_proportions
from amr_predict.plotting import plot_adata
from amr_predict.utils import EmbeddingCache, LinkedDataset, load_as, with_metadata
from fastcluster import linkage, pdist
from loguru import logger
from numpy.random import Generator
from plotnine.ggplot import ggplot
from scipy.cluster.hierarchy import cut_tree
from scipy.spatial.distance import squareform
from scipy.stats import combine_pvalues, entropy, ks_2samp, spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import paired_distances

try:
    from snakemake.script import snakemake as smk
except ImportError:
    smk = type("snakemake", (), {"rule": None, "config": defaultdict(lambda: "")})


RCONFIG = smk.config[smk.rule]
RNG: int = smk.config["rng"]
GEN: Generator = np.random.default_rng(seed=RNG)
EMBEDDING = smk.config["embedding"]
X_KEY = smk.config["pool_embeddings"]["key"]

logger.enable("amr_predict")
logger.add(smk.log[0])


@dataclass
class LongResults:
    metric: list[str] = field(default_factory=lambda: [])
    method: list[str] = field(default_factory=lambda: [])
    value: list[float] = field(default_factory=lambda: [])
    name: list[str] = field(default_factory=lambda: [])
    p_value: list[float] = field(default_factory=lambda: [])

    def from_test(self, test, **kws):
        self.add(p_value=test.pvalue, value=test.statistic, **kws)

    def add(self, metric, method, value, name, p_value):
        self.name.append(name)
        self.p_value.append(p_value)
        self.metric.append(metric)
        self.method.append(method)
        self.value.append(value)


os.environ["HF_HOME"] = smk.config["huggingface"]

# * Utility functions


def record_clustering_metrics(
    fn_dict: dict[str, Callable], y_true, y_pred, results: LongResults, **kwargs
) -> None:
    for metric, fn in fn_dict.items():
        results.metric.append(metric)
        results.value.append(fn(labels_true=y_true, labels_pred=y_pred))
        for k, v in kwargs.items():
            getattr(results, k).append(v)


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
    unrelated: pd.DataFrame = df.groupby(var).sample(
        n=n_pairs_per, replace=replace, random_state=rng
    )
    for group, row in df.groupby(var).agg({id_col: lambda x: list(x)}).iterrows():
        grouped_idx = row[id_col]
        choices = pd.Series(grouped_idx).sample(
            n=n_pairs_per * 2, replace=replace, random_state=rng
        )
        c1 = choices.head(n=n_pairs_per * 2)
        related_pairs.extend([np.array(p) for p in batched(c1, 2)])
        c2 = choices.tail(n=n_pairs_per)

        unrelated_obs = (
            unrelated.loc[unrelated[var] != group, :][id_col]
            .sample(frac=1, random_state=rng)
            .head(n_pairs_per)
        )
        unrelated_pairs.extend([np.array(p) for p in zip(c2, unrelated_obs)])
    return np.array(related_pairs), np.array(unrelated_pairs)


def safe_silhouette_score(**kwargs) -> float:
    try:
        return silhouette_score(**kwargs)
    except ValueError as e:
        logger.warning(f"Warning: ValueError {e} in silhouette score computation")
        return np.nan


def clustering_helper(adata: ad.AnnData, precomputed_dist) -> pl.DataFrame:
    hclust = linkage(precomputed_dist)
    df = adata.obs.copy()
    precomputed = squareform(precomputed_dist)
    results: LongResults = LongResults()
    scoring_metrics = {
        "adjusted_rand": adjusted_rand_score,
        "homogeneity": homogeneity_score,
        "completeness": completeness_score,
    }

    # Gather clustering metrics

    # results["method"].append("leiden")
    # results["metric"].append("fowlkes_mallows")

    # results["value"].append(
    #     safe_silhouette_score(
    #         X=precomputed, labels=raw_results["leiden"], metric="precomputed"
    #     )
    # )
    # results["name"].append("_")

    for y in RCONFIG["cluster_on"]:
        y_true = adata.obs[y]
        n_unique = len(y_true.unique())
        hclust_assignment = cut_tree(hclust, n_clusters=n_unique)
        hclust_assignment[hclust_assignment is None] = -1
        df.loc[:, f"hclust_{y}"] = hclust_assignment

        logger.info("Start recording leiden clustering metrics")
        record_clustering_metrics(
            fn_dict=scoring_metrics,
            y_true=y_true,
            y_pred=df["leiden"],
            results=results,
            name=y,
            method="leiden",
        )
        logger.success("Leiden metrics complete")

        logger.info("Start recording hclust metrics")
        record_clustering_metrics(
            fn_dict=scoring_metrics,
            y_true=y_true,
            y_pred=hclust_assignment[:, 0],
            results=results,
            name=y,
            method="hclust",
        )
        logger.success("Hclust metrics complete")

        # results["method"].append("hclust")
        # results["metric"].append("silhouette")
        # results["value"].append(
        #     safe_silhouette_score(
        #         X=precomputed, labels=hclust_assignment[:, 0], metric="precomputed"
        #     )
        # )
        # results["name"].append(y)

    results.p_value = [-1] * len(results.value)
    return pl.DataFrame(asdict(results))


# * Wrapper functions


def compare_pair_distribution(
    adata: ad.AnnData,
    columns: Sequence,
    distance_metric: str = "cosine",
    **kws,
) -> pl.DataFrame:
    """
    Compare the distribution of distances between related and unrelated sample pairs,
        randomly drawn from `adata`

    A good embedding should have a statistically significantly lower
        distance in the related pairs than unrelated

    The value returned in the result dictionary is the p-value of the comparison
    using the Komolgorov-Smirnov test
    """
    results: LongResults = LongResults()
    kws = kws or {}
    metric = "pair_distribution"
    for col in columns:
        related, unrelated = sample_pairs(adata.obs, col, **kws)
        logger.info(f"Comparing pair distribution for column {col}")
        logger.info(f"n = {related.shape[0]} related pairs")
        logger.info(f"n = {unrelated.shape[0]} unrelated pairs")

        results.add(metric, "n_related_pairs", related.shape[0], col, np.nan)
        results.add(metric, "n_unrelated_pairs", unrelated.shape[0], col, np.nan)

        r1, r2 = adata.X[related[:, 0]], adata.X[related[:, 1]]
        if unrelated.shape[0] != 0:
            u1, u2 = adata.X[unrelated[:, 0]], adata.X[unrelated[:, 1]]
            rel_dist = paired_distances(r1, r2, metric=distance_metric)
            unrel_dist = paired_distances(u1, u2, metric=distance_metric)
            test = ks_2samp(rel_dist, unrel_dist, alternative="greater")  # Alternative
            # is defined in terms of CDFs
            results.from_test(test, metric=metric, method="ks_2samp", name=col)

            results.add(
                metric,
                "kl_div",
                entropy(rel_dist, unrel_dist),
                col,
                np.nan,
            )
        else:
            results.add(metric, "ks_2samp", np.nan, col, np.nan)
            results.add(metric, "kl_div", np.nan, col, np.nan)

        logger.success(f"Pair distribution of {col} complete")
    return pl.DataFrame(
        asdict(results),
        schema={
            "metric": pl.String,
            "method": pl.String,
            "value": pl.Float64,
            "p_value": pl.Float64,
            "name": pl.String,
        },
    )


def covar_dist(
    adata: ad.AnnData,
    columns: Sequence,
    rng: int | None = None,
    distance_metric: str = "cosine",
    seq_start_col: str | None = None,
    seq_end_col: str | None = None,
    seq_id_col: str | None = None,
    n_bootstrap: int | None = None,
) -> pl.DataFrame:
    """Compute the correlation between embedding distances and covariate distances,
    for random pairs of samples
    """
    gen = np.random.default_rng(rng)
    df = adata.obs
    if n_bootstrap:
        pair_mat = gen.choice(
            list(range(adata.shape[0])), (n_bootstrap, 2), replace=True
        )
    correlate_seq_dist = seq_id_col and seq_start_col and seq_end_col
    # TODO: you should probably check which way is better: bootstrapping explicitly
    # or shuffling
    results: LongResults = LongResults()
    if correlate_seq_dist:
        columns.append("_seq_dist_")

    for col in columns:
        logger.info(f"Embedding distance correlation for column {col} started")
        if n_bootstrap is None:
            shuffled = pd.Series(np.arange(adata.shape[0])).sample(
                frac=1, random_state=rng
            )
            if len(shuffled) % 2 != 0:
                shuffled = shuffled[:-1]
            pair_mat = np.array([np.array(p) for p in batched(shuffled, 2)])
        edist = paired_distances(
            adata.X[pair_mat[:, 0]], adata.X[pair_mat[:, 1]], metric=distance_metric
        )
        logger.info("Distance computation complete")
        x = pair_mat[:, 0]
        y = pair_mat[:, 1]
        if correlate_seq_dist and col == "_seq_dist_":
            d1 = df[seq_start_col].values[x] - df[seq_end_col].values[y]
            d2 = df[seq_start_col].values[y] - df[seq_end_col].values[x]
            covar_dist = np.maximum(np.maximum(d1, d2), 0).astype(np.float64)
            covar_dist[df[seq_id_col].values[x] != df[seq_id_col].values[y]] = np.inf
        else:
            covar_dist = np.abs(df[col].iloc[x].values - df[col].iloc[y].values)
        test = spearmanr(edist, covar_dist)
        logger.info("Spearman correlation calculated")
        logger.info(test)
        results.from_test(
            test,
            method=distance_metric,
            metric="covariate_distance_correlation",
            name=col,
        )
        logger.success(f"Routine for column {col} complete")
    return pl.DataFrame(asdict(results))


def comparison_routine(
    dataset_name: str,
    adata: ad.AnnData,
    outdir: Path,
    round: int,
    color_keys: dict,
) -> pl.DataFrame:
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)

    for ck_name, ck in color_keys.items():
        for plot_style in ["pca", "umap"]:
            name = f"{dataset_name}_{plot_style}"
            fig: ggplot = plot_adata(
                adata,
                colors=RCONFIG[ck],
                plot_mode=plot_style,
            )
            fig = fig + gg.ggtitle(title=dataset_name)
            fig.save(outdir / f"plots/{round}_{name}{ck_name}.png", width=15, height=10)

    result_dfs = []
    for metric_group in RCONFIG["methods"]:
        cur_config = RCONFIG.get(metric_group, {})
        cols = cur_config.get("cols", None)
        if cols:
            for col in cols:
                adata.obs[col].fillna("unknown", inplace=True)
        if metric_group == "clustering":
            precomputed = pdist(adata.X, metric="cosine")
            cur_df = clustering_helper(adata=adata, precomputed_dist=precomputed)
        elif metric_group == "neighbor_proportion":
            nn_tmp = nn_proportions(adata, columns=cols, **cur_config["kws"])
            to_concat = []
            for k, v in nn_tmp.items():
                if k.startswith("null"):
                    unpivot_on = ["null_avg", "observed_avg"]
                    if k.endswith("dist"):
                        v = v.with_columns(pl.lit("nn_distance").alias("column"))
                    tmp = (
                        v.unpivot(on=unpivot_on, index=["column", "p_value"])
                        .with_columns(
                            pl.when(pl.col("variable") == "null_avg")
                            .then(None)
                            .otherwise(pl.col("p_value"))
                            .alias("p_value"),
                            pl.lit(f"{k.replace("null", "nn")}").alias("method"),
                        )
                        .rename({"column": "name", "variable": "metric"})
                    )
                    to_concat.append(tmp)
            cur_df = pl.concat(to_concat, how="diagonal_relaxed")
        elif metric_group == "pair_distance_distribution":
            cur_config["kws"]["rng"] = RNG
            cur_df = compare_pair_distribution(
                adata,
                columns=cols,
                distance_metric=cur_config["distance_metric"],
                **cur_config["kws"],
            )
        elif metric_group == "neighbor_preserving_score":
            raise NotImplementedError()

        elif metric_group == "covariate_distance_correlation":
            cur_df = covar_dist(
                adata,
                columns=cols,
                distance_metric=cur_config["distance_metric"],
                rng=RNG,
                seq_id_col=cur_config.get("seq_id_col"),
                seq_start_col=cur_config.get("seq_start_col"),
                seq_end_col=cur_config.get("seq_end_col"),
            )
        result_dfs.append(cur_df)
    return pl.concat(result_dfs, how="diagonal_relaxed")


# * Rules


def run_bootstrap(dset_name: str, adata: ad.AnnData, df_acc: list):
    dfs = []
    for i in range(RCONFIG["bootstrap_rounds"]):
        logger.info(f"Start of bootstrap round {i}")
        logger.info(f"Running with n = {adata.shape[0]} samples")
        if n_samples := RCONFIG.get("n_samples_per"):
            samples = GEN.choice(adata.obs["sample"].unique(), size=n_samples)
            logger.info(f"Subsampled to n = {len(samples)}")
        else:
            samples = adata.obs["sample"].unique()
        subsampled = adata[adata.obs["sample"].isin(samples), :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if smk.rule == "compare_embeddings":
                color_keys = {"": "cluster_on"}
            else:
                color_keys = {"-d": "cluster_on", "-c": "continuous"}
            cur = comparison_routine(
                adata=subsampled,
                dataset_name=dset_name,
                color_keys=color_keys,
                outdir=Path(smk.params["outdir"]),
                round=i,
            )
            dfs.append(cur.with_columns(dataset=pl.lit(dset_name)))
        logger.success(f"Bootstrap round {i} complete")
    if len(dfs) > 1:
        aggregated: pl.DataFrame = pl.concat(dfs)
        aggregated = aggregated.group_by(["metric", "method", "name", "dataset"]).agg(
            pl.col("value").mean().alias("mean"),
            pl.col("value").std().alias("std"),
            pl.col("value").median().alias("median"),
            pl.col("p_value").map_batches(
                lambda x: combine_pvalues(x).pvalue if all(x) != -1 else -1,
                returns_scalar=True,
                return_dtype=pl.Float64,
            ),
        )
        df_acc.append(aggregated)
    else:
        df_acc.extend(dfs)


# ** Embedding comparison


def _compare(is_embeddings: bool = True):
    all_dfs = []
    for d in smk.params["datasets"]:
        dir: Path = Path(d)
        if is_embeddings:
            cache_path = smk.params["caches"] / f"{dir.stem}_{EMBEDDING}_cache"
            cache: EmbeddingCache = EmbeddingCache(cache_path)
            dset: LinkedDataset = cache.to_dataset(load_as(dir, "polars"), "sequence")
            adata = ad.AnnData(X=dset[dset.x_key][:].numpy(), obs=dset.meta.to_pandas())
        else:
            adata = load_as(dir, "adata", x_key=X_KEY)
        if adata.X.shape[1] <= 0:
            logger.debug(adata.X.shape)
            raise ValueError(f"embeddings for {d} have size 0")
        adata = with_metadata(adata, smk.config, "sample", ("sample",))
        adata.obs = adata.obs.replace(
            to_replace={c: np.nan for c in RCONFIG["cluster_on"]}, value="unknown"
        )
        run_bootstrap(dset_name=dir.stem, adata=adata, df_acc=all_dfs)
    pl.concat(all_dfs, how="diagonal_relaxed").write_csv(
        smk.output["metrics"], null_value="NaN"
    )


def compare_embeddings():
    _compare(True)


def compare_pooled():
    _compare(False)


# * Entry
if not (fn := globals().get(smk.rule)):
    raise ValueError(f"Function for rule `{smk.rule}` not defined in this file")
else:
    fn()
