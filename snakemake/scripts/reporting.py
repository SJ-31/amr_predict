#!/usr/bin/env ipython

import anndata as ad
import scanpy as sc
from amr_predict.utils import load_as
from fastcluster import linkage, pdist
from scipy.cluster.hierarchy import cut_tree, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import fowlkes_mallows_score, silhouette_score
from snakemake.script import snakemake as smk

if smk.rule == "compare_embeddings":
    # TODO: give the embedding directory
    for dir in smk.input:
        adata: ad.AnnData = load_as(dir, "adata")
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata)
        raw_results = adata.obs.copy()
        precomputed = pdist(adata.X, metric="cosine")

        # TODO: get the umap, pca and maybe dendrogram

        hclust = linkage(precomputed, metric="prec")
        precomputed = squareform(precomputed)

        results = {"metric": [], "method": [], "score": [], "name": []}
        scoring_metrics = []  # TODO: two metrics here

        # Gather clustering metrics
        results["method"].append("leiden")
        results["metric"].append("fowlkes_mallows")
        results["score"].append(
            silhouette_score(precomputed, labels_pred=raw_results["leiden"])
        )
        results["name"].append("_")

        for y in smk.config["cluster_on"]:
            y_true = adata.obs[y]
            n_unique = len(y_true.unique())
            hclust_assignment = cut_tree(hclust, n_clusters=n_unique)
            raw_results.loc[:, f"hclust_{y}"] = hclust_assignment

            results["method"].append("leiden")
            results["metric"].append("fowlkes_mallows")
            results["score"].append(
                fowlkes_mallows_score(
                    labels_true=y_true, labels_pred=raw_results["leiden"]
                )
            )
            results["name"].append(y)

            results["method"].extend(["hclust", "hclust"])
            results["metric"].extend(["fowlkes_mallows", "silhouette"])
            results["score"].extend(
                [
                    fowlkes_mallows_score(
                        labels_true=y_true, labels_pred=hclust_assignment
                    ),
                    silhouette_score(
                        precomputed, labels=hclust_assignment, metric="precomputed"
                    ),
                ]
            )
            results["name"].append(y)
