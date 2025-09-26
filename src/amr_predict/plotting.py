#!/usr/bin/env ipython

from collections.abc import Iterable, Sequence
from typing import overload

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pypalettes import load_cmap

# TODO: use a different plotting engine for large datasets


def plot_adata(
    adata: ad.AnnData,
    colors: list[str] | str = None,
    subset: Iterable | None = None,
    style: str | None | list[str] = None,
    plot_together: bool = False,
    plot_mode: str = "pca",
    **kwargs,
) -> Figure:
    """Helper function for advanced plotting of adata objects with seaborn

    Parameters
    ----------
    colors : list[str]
        list of columns in adata.obs to color the points by
        The first entry of `colors` is used by subset
    subset : list[str] | None
        Only plot points having class labels in this list
    plot_together : bool
        Whether to include all of the points in the same plot. If False,
        splits the plot up by unique values of colors or y
    style : str | None | list[str]
        Different markers for points
    """
    if "pca" not in adata.uns and plot_mode == "pca":
        sc.pp.pca(adata)
    elif "distances" not in adata.obsp and plot_mode == "umap":
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    show_axes: bool = True
    match plot_mode:
        case "pca":
            obsm_key = "X_pca"
            xlab, ylab = "PC1", "PC2"
            var_ratio = adata.uns["pca"]["variance_ratio"]
            pc1_var, pc2_var = round(var_ratio[0], 2), round(var_ratio[1], 2)
            subtitle = f"PC1, PC2 variance explained: {pc1_var}, {pc2_var}"
        case "umap":
            obsm_key = "X_umap"
            xlab, ylab = "UMAP1", "UMAP2"
            subtitle = "UMAP"
            show_axes = False
        case _:
            raise ValueError(f"Plotting mode {plot_mode} not supported!")

    if isinstance(colors, str):
        colors = [colors]

    y = colors[0]
    keys = adata.obs[y] if subset is None else subset
    ncols = 1 if plot_together else len(keys)
    nrows = len(colors)

    if style is not None and plot_together:
        ncols = len(style)
    elif style is None and plot_together:
        style = [None]
    elif plot_together and isinstance(style, str):
        style = [style]
    elif style is not None and not plot_together and not isinstance(style, str):
        raise ValueError("Multiple styles not supported when !`plot_together`")

    if ncols == 1 and nrows > 1:
        fig, axes = plt.subplots(ncols=nrows, sharey=True, sharex=True)
    else:
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharey=True, sharex=True)

    multiple = ncols > 1
    pts: np.ndarray = adata.obsm[obsm_key]
    data = adata.obs
    if subset is not None and plot_together:
        mask = adata.obs[y].isin(subset)
        type_map = {y: str}
        if style is not None:
            for s in style:
                type_map[s] = str
        data = data.loc[mask, :].astype(type_map)
        pts = pts[mask, :]

    def get_ax(j, i):
        if len(colors) == 1 and not multiple:
            return axes
        elif len(colors) == 1 and multiple:
            return axes[i]
        elif len(colors) > 1 and multiple:
            return axes[j, i]
        elif len(colors) > 1 and not multiple:
            return axes[j]

    def set_labels(ax: Axes):
        if show_axes:
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
        else:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    if not plot_together:
        for j, color in enumerate(colors):
            for i, label in enumerate(keys):
                ax: Axes = get_ax(j, i)
                mask = adata.obs[y] == label
                pt1 = pts[mask, 0]
                pt2 = pts[mask, 1]
                sns.scatterplot(
                    data=data.loc[mask, :],
                    x=pt1,
                    y=pt2,
                    ax=ax,
                    hue=color,
                    style=style,
                    **kwargs,
                )
                set_labels(ax)
                ax.set_title(label)
                if i != len(keys) - 1:
                    ax.get_legend().remove()
    else:
        for j, color in enumerate(colors):
            for i, s in enumerate(style):
                ax: Axes = get_ax(j, i)
                pt1 = pts[:, 0]
                pt2 = pts[:, 1]
                sns.scatterplot(
                    data=data,
                    x=pt1,
                    y=pt2,
                    ax=ax,
                    hue=color,
                    style=s,
                    **kwargs,
                )
                set_labels(ax)
    fig.suptitle(subtitle)
    return fig


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
