#!/usr/bin/env ipython

from collections.abc import Callable
from pathlib import Path

import anndata as ad
import polars as pl
import scanpy as sc
from amr_predict.plotting import plot_adata
from amr_predict.utils import load_as
from fastcluster import linkage, pdist
from scipy.cluster.hierarchy import cut_tree
from scipy.spatial.distance import squareform
from sklearn.metrics import fowlkes_mallows_score, silhouette_score
from snakemake.script import snakemake as smk


def record_clustering_metrics(
    fn_dict: dict[str, Callable], y_true, y_pred, results: dict, **kwargs
) -> None:
    for metric, fn in fn_dict.items():
        results["metric"].append(metric)
        results["score"].append(fn(labels_true=y_true, labels_pred=y_pred))
        for k, v in kwargs.items():
            results[k].append(v)


def comparison_routine(dataset_name: str, adata: ad.AnnData, outdir: Path) -> None:
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    raw_results = adata.obs.copy()
    precomputed = pdist(adata.X, metric="cosine")

    for plot_style in ["pca", "umap"]:
        name = f"{dataset_name}_{plot_style}.png"
        fig = plot_adata(
            adata,
            colors=smk.config["cluster_on"],
            plot_together=True,
            plot_mode=plot_style,
            alpha=0.8,
            legend=False,
        )
        fig.tight_layout()
        fig.set_size_inches((15, 10))
        fig.savefig(outdir.joinpath(name))

    # TODO: if this can't run, use k-means instead
    hclust = linkage(precomputed)
    precomputed = squareform(precomputed)

    results = {"metric": [], "method": [], "score": [], "name": []}
    scoring_metrics = {
        "fowlkes_mallows": fowlkes_mallows_score,
    }

    # Gather clustering metrics

    results["method"].append("leiden")
    results["metric"].append("fowlkes_mallows")

    results["score"].append(silhouette_score(precomputed, labels=raw_results["leiden"]))
    results["name"].append("_")

    for y in smk.config["cluster_on"]:
        y_true = adata.obs[y]
        n_unique = len(y_true.unique())
        hclust_assignment = cut_tree(hclust, n_clusters=n_unique)
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

        results["method"].append("hclust")
        results["metric"].append("silhouette")
        results["score"].append(
            silhouette_score(
                precomputed, labels=hclust_assignment[:, 0], metric="precomputed"
            )
        )
        results["name"].append(y)

    pl.DataFrame(results).write_csv(smk.output["metrics"])


# * Embedding comparison
if smk.rule == "compare_embeddings":
    for dir in smk.params["datasets"]:
        adata: ad.AnnData = load_as(dir, "adata")
        # sc.pp.subsample()  # Maybe subsample if there are too many and it takes too long
        comparison_routine(
            adata=adata, dataset_name=dir.stem, outdir=Path(smk.params["outdir"])
        )
