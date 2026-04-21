#!/usr/bin/env ipython

from collections.abc import Sequence
from functools import reduce
from typing import Literal, overload

import anndata as ad
import plotnine as gg
import polars as pl
import scanpy as sc
from plotnine.ggplot import ggplot
from pypalettes import load_cmap


# TODO: use a different plotting engine for large datasets
def plot_adata(
    adata: ad.AnnData,
    colors: list[str] | str,
    plot_mode: Literal["pca", "umap"] = "pca",
) -> ggplot:
    if "pca" not in adata.uns and plot_mode == "pca":
        sc.pp.pca(adata)
    elif "distances" not in adata.obsp and plot_mode == "umap":
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
    colors = colors if isinstance(colors, list) else [colors]

    if plot_mode == "pca":
        obsm_key = "X_pca"
        xlab, ylab = "PC1", "PC2"
        var_ratio = adata.uns["pca"]["variance_ratio"]
        pc1_var, pc2_var = round(var_ratio[0], 2), round(var_ratio[1], 2)
        subtitle = f"PC1, PC2 variance explained: {pc1_var}, {pc2_var}"
    elif plot_mode == "umap":
        obsm_key = "X_umap"
        xlab, ylab = "UMAP1", "UMAP2"
        subtitle = "UMAP"
    df: pl.DataFrame = pl.concat(
        [
            pl.DataFrame(adata.obsm[obsm_key][:, [0, 1]], schema=[xlab, ylab]),
            pl.from_pandas(adata.obs.loc[:, colors]),
        ],
        how="horizontal",
    )
    plot_multiple = len(colors) > 1
    plots = []
    for i, color_key in enumerate(colors):
        plot = (
            gg.ggplot(df, gg.aes(x=xlab, y=ylab, color=color_key))
            + gg.geom_point()
            + gg.theme(figure_size=(15, 10))
        )
        if plot_mode == "umap":
            plot = plot + gg.theme(
                axis_text_y=gg.element_blank(),
                axis_text_x=gg.element_blank(),
                axis_title_x=gg.element_blank(),
                axis_title_y=gg.element_blank(),
            )
        if i == 0:
            plot = plot + gg.ggtitle(subtitle=subtitle)
        plots.append(plot)
    if len(plots) == 1:
        return plots[0]
    return reduce(lambda x, y: x / y, plots)


@overload
def rand_cmap_d(val: int | None, assign: bool = True) -> list[str]: ...


@overload
def rand_cmap_d(val: Sequence, assign: bool = False) -> dict: ...


def rand_cmap_d(
    val: Sequence | None | int = None, assign: bool = False
) -> list[str] | dict:
    def get_n(length: int) -> list[str]:
        tmp = set()
        while len(tmp) < length:
            tmp |= set(load_cmap().colors)
        return list(tmp)[:length]

    if val is None:
        return load_cmap().colors
    if isinstance(val, int):
        return get_n(val)
    else:
        uniques = set(val)
        colors = get_n(len(uniques))
        mapping = dict(zip(uniques, colors))
        if assign:
            return [mapping[v] for v in val]
        return mapping
