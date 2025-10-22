#!/usr/bin/env ipython

import os
import warnings
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
from amr_predict.plotting import plot_adata
from amr_predict.utils import load_as, vecdist
from fastcluster import linkage, pdist
from numpy.random import Generator
from plotnine.ggplot import ggplot
from scipy.cluster.hierarchy import cut_tree
from scipy.spatial.distance import squareform
from scipy.stats import entropy, ks_2samp, spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

try:
    from snakemake.script import snakemake as smk
except ImportError:
    smk = type("snakemake", (), {"rule": None, "config": {}})

RCONFIG = smk.config[smk.rule]
RNG: int = smk.config["rng"]


@dataclass
class LongResults:
    metric: list[str] = field(default_factory=lambda: [])
    method: list[str] = field(default_factory=lambda: [])
    value: list[float] = field(default_factory=lambda: [])
    name: list[float] = field(default_factory=lambda: [])


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
        print(f"Warning: ValueError {e} in silhouette score computation")
        return np.nan


def clustering_helper(adata: ad.AnnData, precomputed_dist) -> LongResults:
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

        record_clustering_metrics(
            fn_dict=scoring_metrics,
            y_true=y_true,
            y_pred=df["leiden"],
            results=results,
            name=y,
            method="leiden",
        )

        record_clustering_metrics(
            fn_dict=scoring_metrics,
            y_true=y_true,
            y_pred=hclust_assignment[:, 0],
            results=results,
            name=y,
            method="hclust",
        )

        # results["method"].append("hclust")
        # results["metric"].append("silhouette")
        # results["value"].append(
        #     safe_silhouette_score(
        #         X=precomputed, labels=hclust_assignment[:, 0], metric="precomputed"
        #     )
        # )
        # results["name"].append(y)

    return results


def nn_proportions(
    adata: ad.AnnData, columns: Sequence, **kws
) -> tuple[pl.DataFrame, pl.DataFrame]:
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
    tuple of two dataframes
        The first dataframe is of shape (n_samples, len(columns)). Each column is the
        proportion of the sample's nearest neighbors that were the same under that
        column name

        The second dataframe contains summary statistics for neighbor distances for each
        sample
    """
    nclass = NearestNeighbors(**kws)
    nclass.fit(adata.X)
    distances, neighbors = nclass.kneighbors()
    df = adata.obs
    n_neighbors = neighbors.shape[1]
    tmp = {}
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
    dfs = [
        pl.concat([index_df, pl.DataFrame(tmp)], how="horizontal"),
        pl.concat(
            [index_df, pl.concat([from_horizontal, from_transposed], how="horizontal")],
            how="horizontal",
        ),
    ]
    return dfs[0], dfs[1]


# * Wrapper functions


def compare_pair_distribution(
    adata: ad.AnnData,
    columns: Sequence,
    distance_metric: str = "cosine",
    **kws,
) -> LongResults:
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
    for col in columns:
        related, unrelated = sample_pairs(adata.obs, col, **kws)
        r1, r2 = adata.X[related[:, 0]], adata.X[related[:, 1]]
        u1, u2 = adata.X[unrelated[:, 0]], adata.X[unrelated[:, 1]]
        rel_dist = vecdist(r1, r2, metric=distance_metric)
        unrel_dist = vecdist(u1, u2, metric=distance_metric)
        results.metric.append("pair_distribution")
        results.method.append("ks_2samp_pval")
        test = ks_2samp(rel_dist, unrel_dist, alternative="greater")  # Alternative
        # is defined in terms of CDFs
        results.value.append(test.pvalue)
        results.name.append(col)

        results.metric.append("pair_distribution")
        results.method.append("kl_div")
        results.value.append(entropy(rel_dist, unrel_dist))
        results.name.append(col)
    return results


def covar_dist(
    adata: ad.AnnData,
    columns: Sequence,
    rng: int | None = None,
    distance_metric: str = "cosine",
) -> LongResults:
    results: LongResults = LongResults()
    for col in columns:
        shuffled = pd.Series(np.arange(adata.shape[0])).sample(frac=1, random_state=rng)
        if len(shuffled) % 2 != 0:
            shuffled = shuffled[:-1]
        pair_mat = np.array([np.array(p) for p in batched(shuffled, 2)])
        edist = vecdist(adata.X[pair_mat[:, 0]], adata.X[pair_mat[:, 1]], "cosine")
        covar_dist = np.abs(
            adata.obs[col].iloc[pair_mat[:, 0]].values
            - adata.obs[col].iloc[pair_mat[:, 1]].values
        )
        test = spearmanr(edist, covar_dist)
        results.method.append(distance_metric)
        results.metric.append("covariate_distance_correlation")
        results.name.append(col)
        results.value.append(test.statistic)

        results.method.append(distance_metric)
        results.metric.append("covariate_distance_correlation_pval")
        results.name.append(col)
        results.value.append(test.pvalue)
    return results


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
        if metric_group == "clustering":
            precomputed = pdist(adata.X, metric="cosine")
            lr = clustering_helper(adata=adata, precomputed_dist=precomputed)
            result_dfs.append(pl.DataFrame(asdict(lr)))
        elif metric_group == "neighbor_proportion":
            nprop, dist_stats = nn_proportions(adata, columns=cols, **cur_config["kws"])
            prop_agg = (
                nprop.drop("index")
                .mean()
                .unpivot(variable_name="name")
                .with_columns(
                    pl.lit("avg_nn_proportion").alias("metric"),
                    pl.lit(cur_config["kws"]["metric"]).alias("method"),
                )
                .select(["metric", "method", "value", "name"])
            )
            result_dfs.append(prop_agg)
            result_dfs.append(
                pl.DataFrame(
                    {
                        "metric": "avg_nn_distance",
                        "method": "max",
                        "value": dist_stats["max"].mean(),
                        "name": "-",
                    }
                )
            )
        elif metric_group == "pair_distance_distribution":
            cur_config["kws"]["rng"] = RNG
            lr = compare_pair_distribution(
                adata,
                columns=cols,
                distance_metric=cur_config["distance_metric"],
                **cur_config["kws"],
            )
            result_dfs.append(pl.DataFrame(asdict(lr)))
        elif metric_group == "covariate_distance_correlation":
            lr = covar_dist(
                adata,
                columns=cols,
                distance_metric=cur_config["distance_metric"],
                rng=RNG,
            )
            result_dfs.append(pl.DataFrame(asdict(lr)))
    return pl.concat(result_dfs)


# * Rules

# ** Embedding comparison
if smk.rule in {"compare_embeddings", "compare_pooled"}:
    rng: Generator = np.random.default_rng(seed=smk.config["rng"])
    all_dfs = []
    for dir in smk.params["datasets"]:
        adata: ad.AnnData = load_as(dir, "adata")
        adata.obs = adata.obs.replace(
            to_replace={c: np.nan for c in RCONFIG["cluster_on"]}, value="unknown"
        )
        dfs = []
        for i in range(RCONFIG["bootstrap_rounds"]):
            if n_samples := RCONFIG.get("n_samples_per"):
                samples = rng.choice(adata.obs["sample"].unique(), size=n_samples)
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
                    dataset_name=dir.stem,
                    color_keys=color_keys,
                    outdir=Path(smk.params["outdir"]),
                    round=i,
                )
                dfs.append(cur)
        aggregated: pl.DataFrame = pl.concat(dfs)
        aggregated = (
            aggregated.group_by(["metric", "method", "name"])
            .agg(
                pl.col("value").mean().alias("mean"),
                pl.col("value").std().alias("std"),
                pl.col("value").median().alias("median"),
            )
            .with_columns(dataset=pl.lit(dir.stem))
        )
        all_dfs.append(aggregated)
    pl.concat(all_dfs).write_csv(smk.output["metrics"])
