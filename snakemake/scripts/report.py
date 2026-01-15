#!/usr/bin/env ipython

from pathlib import Path
from typing import Literal

import matplotlib
import plotnine as gg
import polars as pl
from loguru import logger
from plotnine.helpers import get_aesthetic_limits

matplotlib.use("QtAgg")
# BUG: [2026-01-14 Wed] due to issues with Tkinter backend on local
# Needs PyQt5 installed

try:
    from snakemake.script import snakemake as smk
except ImportError:
    smk = type("snakemake", (), {"rule": None, "config": {}, "log": [0]})

RCONFIG = smk.config.get(smk.rule)
RNG: int = smk.config.get("rng")
logger.enable("amr_predict")

# logger.add(smk.log[0])

CONFIG = smk.config


# * Plotting functions

# ** Compare embeddings
# In the functions below, "metrics" is the dataframe obtained by reading
# "metrics.csv" produced by compare_embeddings


def nn_plot(
    metrics: pl.DataFrame,
    batch_value: Literal["mean", "median"] = "mean",
    composition_size: tuple[int, int] = (20, 10),
) -> dict:
    df = metrics.filter(pl.col("method").str.starts_with("nn_")).with_columns(
        pl.col("p_value")
        .cast(pl.String)
        .replace("NaN", "")
        .map_elements(lambda x: f"p: {x}" if x else x, return_dtype=pl.String),
        pl.col("metric").replace_strict(
            {"observed_avg": "Observed", "null_avg": "Null"}
        ),
    )
    is_bootstrap: bool = "mean" in metrics.columns
    dist_plot: gg.ggplot | None = None
    y_col = batch_value if is_bootstrap else "value"

    def plot_helper(
        cur: pl.DataFrame,
        title: str,
        y_lab: str,
        facet: str | None = None,
    ) -> gg.ggplot:
        plot = (
            gg.ggplot(cur, gg.aes(x="metric", y=y_col, fill="dataset"))
            + gg.geom_col(stat="identity", position="dodge")
            + gg.geom_label(
                gg.aes(label="p_value"), position=gg.position_dodge(width=0.9)
            )
        )
        if is_bootstrap:
            plot = plot + gg.geom_errorbar(gg.aes(ymin="-std", ymax="std"))
        if facet is not None:
            plot = plot + gg.facet_wrap(facet)
        subtitle = f"{batch_value} across bootstrap samples" if is_bootstrap else None
        return (
            plot
            + gg.ggtitle(title, subtitle=subtitle)
            + gg.xlab("Neighbor distribution")
            + gg.ylab(y_lab)
        )

    if "nn_distance" in df["name"]:
        nn_dist = df.filter(pl.col("name") == "nn_distance")
        df = df.filter(pl.col("name") != "nn_distance")
        dist_plot = plot_helper(nn_dist, "Average distance between neighbors", "Value")
    prop_plot = plot_helper(
        df.filter(pl.col("method") == "nn_prop"),
        "Proportion of neighbors with the same label",
        "Proportion",
        facet="name",
    )
    imp_plot = plot_helper(
        df.filter(pl.col("method") == "nn_impurity").with_columns(pl.col("value")),
        "Gini impurity by labels",
        "Impurity",
        facet="name",
    )
    result = {"nn_prop": prop_plot, "nn_impurity": imp_plot}
    if dist_plot is not None:
        result["composed"] = dist_plot | (
            (
                prop_plot
                + gg.theme(axis_title_x=gg.element_blank(), legend_position="none")
            )
            / (imp_plot + gg.theme(legend_position="none"))
        )
        result["nn_dist"] = dist_plot
    else:
        result["composed"] = (prop_plot + gg.theme(legend_position=None)) | imp_plot
    result["composed"] = result["composed"] + gg.theme(figure_size=composition_size)
    return result


def covar_dist_plot(
    metrics: pl.DataFrame, raw: pl.DataFrame, log_x: bool = True
) -> gg.ggplot:
    df = raw.join(
        metrics.filter(pl.col("metric") == "covariate_distance_correlation"),
        on=("name", "dataset"),
        how="left",
    ).with_columns(
        pl.struct(["p_value", "value"])
        .map_elements(
            lambda x: f"{round(x["value"], 3)} (p: {round(x['p_value'], 3)})",
            return_dtype=pl.String,
        )
        .alias("annotation")
    )
    dm: str = df["method"].first()
    title = "Relationship between embedding and covariate distance"
    subtitle = f"Spearman correlation, with {dm} distance"
    corr_plot = (
        gg.ggplot(df, gg.aes(x="embedding_distance", y="covariate_distance"))
        + gg.geom_point()
        + gg.ggtitle(title, subtitle)
        + gg.facet_grid("name ~ dataset")
        + gg.xlab("Embedding distance")
        + gg.ylab("Covariate distance")
        + gg.geom_text(
            gg.aes(x=0, y=0, label="annotation"),
            stat="unique",
            ha="left",
            alpha=0.7,
        )
    )
    if log_x:
        corr_plot = (
            corr_plot
            + gg.scale_x_log10()
            + gg.theme(panel_grid_minor_x=gg.element_blank())
        )
    return corr_plot


def pair_dist_plot(
    metrics: pl.DataFrame, raw: pl.DataFrame, bins: int = 15, log_x: bool = True
) -> gg.ggplot:
    df = raw.join(
        metrics.filter(
            (pl.col("metric") == "pair_distribution")
            & (~pl.col("method").str.ends_with("pairs"))
        ).pivot(on="method", values=["value", "p_value"]),
        on=["name", "dataset"],
        how="left",
    ).with_columns(
        pl.struct(["value_kl_div", "p_value_ks_2samp"])
        .map_elements(
            lambda x: f"{round(x["value_kl_div"], 3)} (p: {round(x["p_value_ks_2samp"], 3)})",
            return_dtype=pl.String,
        )
        .alias("annotation")
    )
    plot = (
        gg.ggplot(df, gg.aes(x="value", fill="group"))
        + gg.geom_histogram(bins=bins)
        + gg.facet_grid("name ~ dataset")
    )
    ylim = max([max(lim) for lim in get_aesthetic_limits(plot, "y")])
    plot = plot + gg.geom_text(
        gg.aes(label="annotation", y=ylim, x=0),
        stat="unique",
        ha="left",
        va="top",
        alpha=0.7,
    )
    if log_x:
        plot = (
            plot + gg.scale_x_log10() + gg.theme(panel_grid_minor_x=gg.element_blank())
        )
    return plot


def cluster_metric_plot(metrics: pl.DataFrame) -> gg.ggplot:
    df = metrics.filter(pl.col("method").is_in(("leiden", "hclust")))
    plot = (
        gg.ggplot(df, gg.aes(x="method", y="value", fill="dataset"))
        + gg.geom_col(stat="identity", position="dodge")
        + gg.facet_grid("metric ~ name")
    )
    return plot



# ** Evaluation


def evaluation_plots():
    groups, ttypes = ("cv", "holdout", "ctrl_cv"), ("regression", "classification")
    for group in groups:
        outdir = smk.params["outdir"] / f".{group}"
        outdir.mkdir(exist_ok=True)
        for task in ttypes:
            key = f"{group}_{task[0]}"
            if key not in smk.input.keys():
                continue
            combined: pl.DataFrame = pl.concat(
                [
                    pl.read_csv(csv).with_columns(
                        pl.lit(csv.stem.removesuffix(f"_{task}")).alias("dataset"),
                        pl.lit(csv.parent.stem).alias("model"),
                    )
                    for csv in (Path(f) for f in smk.input[key])
                ]
            )
            metrics = combined["metric"].unique()
            for metric in metrics:
                metric_outfile = outdir / f"{metric}_{task}.png"
                filtered = combined.filter(pl.col("metric") == metric)
                bplots = (
                    gg.ggplot(filtered, gg.aes(x="task", y="value", fill="dataset"))
                    + gg.geom_boxplot()
                    + gg.facet_wrap("model")
                )
                bplots.save(metric_outfile, **CONFIG["plotnine"]["small"])
            # TODO: generate aggregated files for datavzrd
            # TODO:
            # agg = combined.group_by(["dataset", "model", "task", "metric"]).agg(
            #     pl.col("value").mean().alias("mean"),
            #     pl.col("value").median().alias("median"),
            #     pl.col("value").std().alias("std"),
            # )


def embedding_comparison():
    metrics: pl.DataFrame = pl.read_csv(smk.input)
    # Will need to make bespoke tables or plots for each metric group
    # e.g. pair distribution can be barcharts faceted by method (after removing
    # ks_2samp). name should be x, value y, dataset the fill
    # for j
