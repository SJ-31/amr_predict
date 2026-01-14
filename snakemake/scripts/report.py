#!/usr/bin/env ipython

from pathlib import Path
from typing import Literal

import matplotlib
import plotnine as gg
import polars as pl
from loguru import logger

matplotlib.use("QtAgg")

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
):
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


def pair_dist_plot(metrics: pl.DataFrame):
    df = metrics.filter(pl.col("metric") == "pair_distribution")


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
