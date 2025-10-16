#!/usr/bin/env ipython

import os
import warnings
from collections.abc import Callable
from pathlib import Path

import plotnine as gg
from plotnine.ggplot import ggplot
from snakemake.script import snakemake as smk

os.environ["HF_HOME"] = smk.config["huggingface"]

import anndata as ad
import numpy as np
import polars as pl
import scanpy as sc
from amr_predict.plotting import plot_adata
from amr_predict.utils import load_as
from fastcluster import linkage, pdist
from numpy.random import Generator
from scipy.cluster.hierarchy import cut_tree
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    silhouette_score,
)


def record_clustering_metrics(
    fn_dict: dict[str, Callable], y_true, y_pred, results: dict, **kwargs
) -> None:
    for metric, fn in fn_dict.items():
        results["metric"].append(metric)
        results["value"].append(fn(labels_true=y_true, labels_pred=y_pred))
        for k, v in kwargs.items():
            results[k].append(v)


def safe_silhouette_score(**kwargs) -> float:
    try:
        return silhouette_score(**kwargs)
    except ValueError as e:
        print(f"Warning: ValueError {e} in silhouette score computation")
        return np.nan


def comparison_routine(
    dataset_name: str,
    adata: ad.AnnData,
    outdir: Path,
    config: dict,
    round: int,
    color_keys: dict,
) -> pl.DataFrame:
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    raw_results = adata.obs.copy()
    precomputed = pdist(adata.X, metric="cosine")

    for ck_name, ck in color_keys.items():
        for plot_style in ["pca", "umap"]:
            name = f"{dataset_name}_{plot_style}.png"
            fig: ggplot = plot_adata(
                adata,
                colors=config[ck],
                plot_mode=plot_style,
            )
            fig = fig + gg.ggtitle(title=dataset_name)
            fig.save(outdir / f"plots/{round}_{name}{ck_name}", width=15, height=10)

    # TODO: if this can't run, use k-means instead
    hclust = linkage(precomputed)
    precomputed = squareform(precomputed)

    results = {"metric": [], "method": [], "value": [], "name": []}
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

    for y in config["cluster_on"]:
        y_true = adata.obs[y]
        n_unique = len(y_true.unique())
        hclust_assignment = cut_tree(hclust, n_clusters=n_unique)
        hclust_assignment[hclust_assignment is None] = -1
        raw_results.loc[:, f"hclust_{y}"] = hclust_assignment

        record_clustering_metrics(
            fn_dict=scoring_metrics,
            y_true=y_true,
            y_pred=raw_results["leiden"],
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

    return pl.DataFrame(results)


# * Embedding comparison
if smk.rule in {"compare_embeddings", "compare_pooled"}:
    config = smk.config[smk.rule]
    rng: Generator = np.random.default_rng(seed=smk.config["rng"])
    all_dfs = []
    for dir in smk.params["datasets"]:
        adata: ad.AnnData = load_as(dir, "adata")
        adata.obs = adata.obs.replace(
            to_replace={c: np.nan for c in config["cluster_on"]}, value="unknown"
        )
        dfs = []
        for i in range(config["bootstrap_rounds"]):
            if n_samples := config.get("n_samples_per"):
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
                    config=config,
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
