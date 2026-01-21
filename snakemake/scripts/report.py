#!/usr/bin/env ipython

import re
import shutil
from pathlib import Path
from subprocess import run
from typing import Literal

import matplotlib
import plotly.express as px
import plotnine as gg
import polars as pl
import polars.selectors as cs
import yaml
from amr_predict.utils import load_as, plot_params, with_metadata
from loguru import logger
from plotly.graph_objs._figure import Figure
from plotnine.helpers import get_aesthetic_limits

matplotlib.use("QtAgg")
# BUG: [2026-01-14 Wed] due to issues with Tkinter backend on local
# Needs PyQt5 installed

try:
    from snakemake.script import snakemake as smk
except ImportError:
    smk = type(
        "snakemake", (), {"rule": "none", "config": {}, "log": [0], "params": {}}
    )

RNG: int = smk.config.get("rng")
logger.enable("amr_predict")

# logger.add(smk.log[0])

CONFIG = smk.config


# * Plotting functions


def safe_round(val, to: int = 3):
    if isinstance(val, str):
        val = float(val)
    if val is not None:
        return round(val, to)
    return val


# ** Compare embeddings
# In the functions below, "metrics" is the dataframe obtained by reading
# "metrics.csv" produced by compare_embeddings


def nn_plot(
    metrics: pl.DataFrame,
    batch_value: Literal["mean", "median"] = "mean",
) -> dict:
    df = metrics.filter(pl.col("method").str.starts_with("nn_")).with_columns(
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
        wrap: bool = False,
    ) -> gg.ggplot:
        plot = gg.ggplot(
            cur, gg.aes(x="metric", y=y_col, fill="dataset", color="metric")
        ) + gg.geom_col(stat="identity", position="dodge")
        if is_bootstrap:
            plot = plot + gg.geom_errorbar(
                gg.aes(y=y_col, ymin=f"{y_col}-std", ymax=f"{y_col}+std"), color="black"
            )
        if facet is not None and not wrap:
            plot = plot + gg.facet_grid(facet)
        elif facet is not None:
            plot = plot + gg.facet_wrap(facet)
        subtitle = (
            f"{batch_value} across bootstrap samples, bars show std"
            if is_bootstrap
            else None
        )
        return (
            plot
            + gg.ggtitle(title, subtitle=subtitle)
            + gg.xlab("Neighbor distribution")
            + gg.ylab(y_lab)
            + gg.guides(fill="none", color="none")
            + gg.geom_label(
                gg.aes(label="p_value", y=float("inf"), x=0),
                stat="unique",
                color="black",
                ha="left",
                va="top",
                format_string="  {}",
            )
        )

    if "nn_distance" in df["name"]:
        nn_dist = df.filter(pl.col("name") == "nn_distance")
        df = df.filter(pl.col("name") != "nn_distance")
        dist_plot = plot_helper(
            nn_dist,
            "Average distance between neighbors",
            "Value",
            facet="dataset",
            wrap=True,
        )
    prop_plot = plot_helper(
        df.filter(pl.col("method") == "nn_prop"),
        "Proportion of neighbors with the same label",
        "Proportion",
        facet="name ~ dataset",
    )
    imp_plot = plot_helper(
        df.filter(pl.col("method") == "nn_impurity").with_columns(pl.col(y_col)),
        "Gini impurity by labels",
        "Impurity",
        facet="name ~ dataset",
    )
    result = {
        "nn_proportion": prop_plot,
        "nn_impurity": imp_plot,
        "nn_distance": dist_plot,
    }
    return result


def covar_dist_plot(
    metrics: pl.DataFrame,
    raw: pl.DataFrame,
    log_x: bool = True,
    batch_value: Literal["mean", "median"] = "mean",
) -> gg.ggplot:
    y_col = batch_value if "mean" in metrics.columns else "value"
    df = raw.join(
        metrics.filter(pl.col("metric") == "covariate_distance_correlation"),
        on=("name", "dataset"),
        how="left",
    ).with_columns(
        pl.struct(["p_value", y_col])
        .map_elements(
            lambda x: f"{safe_round(x[y_col])} ({x['p_value']})",
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
    metrics: pl.DataFrame,
    raw: pl.DataFrame,
    bins: int = 15,
    log_x: bool = True,
    batch_value: Literal["mean", "median"] = "mean",
) -> gg.ggplot:
    value = batch_value if "mean" in metrics.columns else "value"
    df = raw.rename({"value": "raw_value"}).join(
        metrics.filter(
            (pl.col("metric") == "pair_distribution")
            & (~pl.col("method").str.ends_with("pairs"))
        ),
        on=["name", "dataset"],
        how="left",
    )
    p_val_lookup: dict = {
        k: v[0]["p_value"]
        for k, v in df.filter(pl.col("method") == "ks_2samp")
        .rows_by_key(["name", "dataset"], named=True)
        .items()
    }
    df = df.filter(pl.col("method") == "kl_div").with_columns(
        pl.struct([value, "name", "dataset"])
        .map_elements(
            lambda x: f"{safe_round(x[value])} ({p_val_lookup[(x["name"], x["dataset"])]})",
            return_dtype=pl.String,
        )
        .alias("annotation")
    )
    plot = (
        gg.ggplot(df, gg.aes(x="raw_value", fill="group"))
        + gg.geom_histogram(bins=bins)
        + gg.facet_grid("name ~ dataset")
    )
    ylim = max(
        [
            max(lim) if isinstance(lim, tuple) else lim
            for lim in get_aesthetic_limits(plot, "y")
        ]
    )
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


def cluster_metric_plot(
    metrics: pl.DataFrame, batch_value: Literal["mean", "median"] = "mean"
) -> gg.ggplot:
    value = batch_value if "mean" in metrics.columns else "value"
    df = metrics.filter(pl.col("method").is_in(("leiden", "hclust")))
    plot = (
        gg.ggplot(df, gg.aes(x="method", y=value, fill="dataset"))
        + gg.geom_col(stat="identity", position="dodge")
        + gg.facet_grid("metric ~ name")
    )
    return plot


# * Rule functions


def format_metadata(cfg: dict = CONFIG):
    df: pl.DataFrame = load_as(smk.input[0], "polars").select("sample")
    all_tasks = [t for tasks in cfg["tasks"].values() for t in tasks]
    cfg["ast_metadata"]["binarize"] = True
    ast_binarized = (
        with_metadata(df, cfg, "sample", "ast")
        .select(["sample"] + all_tasks)
        .rename(lambda x: x.replace("_class", "_resistant"))
        .select("sample", cs.ends_with("_resistant"))
    )
    cfg["ast_metadata"]["binarize"] = False
    ast = (
        with_metadata(df, cfg, "sample", "ast")
        .select(["sample"] + all_tasks)
        .with_columns(
            cs.ends_with("_class").replace_strict(
                {"resistant": "R", "intermediate": "I", "susceptible": "S"}
            )
        )
        .join(ast_binarized, on="sample")
    )
    ast.write_csv(smk.output["ast"], null_value="NA")
    sample_meta = with_metadata(df, cfg, "sample", "sample")
    for col in smk.params["to_remove"]:
        if col in sample_meta.columns:
            sample_meta = sample_meta.drop(col)
    sample_meta.write_csv(smk.output["meta"], null_value="NA")


def plot_eval(
    df: pl.DataFrame, task, outdir, method, plotly: bool = False, cfg: dict = CONFIG
):
    metrics = df["metric"].unique()
    if plotly:
        raise NotImplementedError()
    for metric in metrics:
        metric_outfile = outdir / f"{metric}_{task}.svg"
        filtered = df.filter(pl.col("metric") == metric)
        plots = gg.ggplot(filtered, gg.aes(x="task", y="value", fill="dataset"))
        if method == "holdout":
            plots = (
                plots
                + gg.geom_bar(stat="identity", position="dodge")
                + gg.facet_grid("model ~ test_set")
            )
        else:
            plots = plots + gg.geom_boxplot() + gg.facet_wrap("model")
        plots.save(metric_outfile, **plot_params("evaluation", cfg))


def evaluation():
    eval_methods, ttypes = (
        ("cv", "holdout", "ctrl_cv"),
        ("regression", "classification"),
    )
    out = Path(smk.params["outdir"])
    for method in eval_methods:
        if not (out / method).exists():
            continue
        outdir = out / f".{method}"
        outdir.mkdir(exist_ok=True)
        for task in ttypes:
            key = f"{method}_{task[0]}"
            if key not in smk.params.keys():
                continue
            combined: pl.DataFrame = pl.concat(
                [
                    pl.read_csv(csv).with_columns(
                        pl.lit(csv.stem.removesuffix(f"_{task}")).alias("dataset"),
                        pl.lit(csv.parent.stem).alias("model"),
                    )
                    for csv in (Path(f) for f in smk.params[key])
                ]
            )
            plot_eval(combined, task=task, outdir=outdir, method=method)

            # TODO: generate aggregated files for datavzrd
            # TODO:
            # agg = combined.group_by(["dataset", "model", "task", "metric"]).agg(
            #     pl.col("value").mean().alias("mean"),
            #     pl.col("value").median().alias("median"),
            #     pl.col("value").std().alias("std"),
            # )


def embedding_comparison():
    dir = Path(smk.input[0])
    outdir = Path(smk.output[0])
    outdir.mkdir(exist_ok=True)
    all_metrics: pl.DataFrame = pl.concat(
        [pl.read_csv(file) for file in dir.glob("*metrics.csv")]
    ).with_columns(
        pl.col("p_value")
        .cast(pl.String)
        .replace("NaN", "")
        .map_elements(
            lambda x: f"p: {safe_round(x)}" if x else x, return_dtype=pl.String
        ),
    )
    try:
        for method in smk.config[smk.params["rule"]]["methods"]:
            save_file = outdir / f"{method}.svg"
            if method == "neighbor_proportion":
                nn: dict = nn_plot(all_metrics)
                for k, v in nn.items():
                    v.save(
                        outdir / f"{k}.svg",
                        **plot_params("embedding_comparison", CONFIG),
                    )
                continue
            elif method == "covariate_distance_correlation":
                plot = covar_dist_plot(
                    all_metrics,
                    pl.concat(
                        [
                            pl.read_csv(file)
                            for file in dir.glob("*covariate_distance*.csv")
                        ]
                    ),
                    log_x=True,
                )
            elif method == "pair_distance_distribution":
                plot = pair_dist_plot(
                    all_metrics,
                    pl.concat(
                        [pl.read_csv(file) for file in dir.glob("*pair_distance*.csv")]
                    ),
                    bins=20,
                    log_x=True,
                )
            elif method == "clustering":
                plot = cluster_metric_plot(all_metrics)
            else:
                raise ValueError("method not recognized")
            plot.save(save_file, **plot_params("embedding_comparison", CONFIG))
    except Exception as e:
        shutil.rmtree(outdir)
        raise e


def record_env(
    with_rich: bool = True,
    outdir: Path | None = None,
    cfg: dict = CONFIG,
    rich_kws: dict | None = None,
):
    """Write relevant parts of the analysis configuration to separate yaml
        files to display

    Parameters
    ----------
    with_rich : bool
        Whether to use rich-cli to produce formatted versions of the config
        groups, with syntax highlighting


    Notes
    -----
    This only includes configuration items relevant to interpretation of results i.e.
        file paths, plotting parameters, renaming rules etc. will be excluded

    """
    outdir = outdir or Path(smk.params["outdir"])
    outdir.mkdir(exist_ok=True)
    rich_kws = rich_kws or CONFIG.get("report_plots", {}).get("rich-cli") or {}
    to_save: dict[str | tuple, dict[str, str | None]] = {
        "evaluation": {
            "models": "model_parameters",
            "baseline_filtering": None,
            "trainer": "lightning_trainer",
            "dataloader": "torch_dataloader",
            "cross_validate": "cross_validation",
            "holdout": None,
            "tasks": None,
        },
        "embedding_interpretation": {
            "train_sae": "SAE_training",
            ("eval_sae", "umap"): "latent_umap",
        },
        "embedding_parameters": {
            ("embedding_methods", CONFIG["embedding"]): CONFIG["embedding"]
        },
        "data_preparation": {
            ("pool_embeddings", "methods"): "pooling methods",
            "preprocessing": "sequence_preprocessing",
        },
        "embedding_comparison": {
            "compare_embeddings": "sequence_embeddings",
            "compare_pooled": "pooled_genome_embeddings",
        },
    }
    # mapping of config groups to dictionary describing items. The dictionary
    # has the format {"CONFIG_KEY": "OPTIONAL_SYNONYM"}
    # If the CONFIG_KEY is a tuple, then descend sequentially through the keys listed
    saved_groups: dict[str, dict] = {}
    for group_name, items in to_save.items():
        saved_groups[group_name] = {}
        for k, v in items.items():
            if isinstance(k, tuple):
                to_iter = iter(k)
                item = cfg[next(to_iter)]
                for nt in to_iter:
                    item = item[nt]
            else:
                item = cfg[k]
            key = v if v else group_name
            saved_groups[group_name][key] = item
    for k, v in saved_groups.items():
        saved_file = outdir / f"{k}.yaml"
        with open(saved_file, "w") as f:
            yaml.safe_dump(v, f)
        if with_rich:
            html_file = outdir / f"{k}.html"
            command = f"rich {saved_file} -o {html_file}"
            for arg, val in rich_kws.items():
                if val is not None:
                    command = f"{command} --{arg} {val}"
                else:
                    command = f"{command} --{arg}"
            proc = run(command, shell=True, capture_output=True)
            proc.check_returncode()
            if not html_file.exists():
                raise ValueError(f"rich failed to produce html file {html_file}")


def eval_sae():
    dfs: dict[str, pl.DataFrame] = {}
    for suffix in ("latent_counts", "concept_scores"):
        cur = pl.concat(
            [
                pl.read_csv(file).with_columns(
                    pl.lit(re.findall(f"\\.(.*)_{suffix}", file.stem)[0]).alias(
                        "dataset"
                    ),
                )
                for file in Path(smk.input[0]).glob(f".*_{suffix}.csv")
            ]
        )
        dfs[suffix] = cur
        cur.write_csv(smk.output[suffix])

    lc_df = dfs["latent_counts"].with_columns(
        (pl.col("count") / pl.col("count").sum())
        .over(["activation_source", "dataset"])
        .alias("frac")
    )
    lc_plot = (
        gg.ggplot(lc_df, gg.aes(x="activation_source", y="frac", fill="type"))
        + gg.geom_col(position="dodge", stat="identity")
        + gg.facet_wrap("dataset")
        + gg.xlab("Activation source")
        + gg.ylab("Fraction")
        + gg.ggtitle("Latent categories per dataset")
    )
    lc_plot.save(smk.output["frac_plot"], **plot_params("sae_latent_fractions", CONFIG))

    score_dir = Path(smk.output["score_plot"])
    score_dir.mkdir(exist_ok=True)
    metric_cols = [
        col
        for col in dfs["concept_scores"].columns
        if col
        not in {
            "dataset",
            "level",
            "concept",
            "cluster",
            "latent_idx",
            "activation_source",
            "label_max",
        }
    ]
    score_df = dfs["concept_scores"].unpivot(
        index=["latent_idx", "activation_source", "concept", "dataset"], on=metric_cols
    )
    classification_metrics = [
        "sensitivity",
        "precision",
        "specificity",
        "f1",
        "silhouette_samples",
    ]
    for concept in score_df["concept"].unique():
        splot = (
            gg.ggplot(
                score_df.filter(
                    (pl.col("variable").is_in(classification_metrics))
                    & (pl.col("concept") == concept)
                ),
                gg.aes(y="value", fill="dataset"),
            )
            + gg.geom_boxplot()
            + gg.facet_grid("variable ~ activation_source ", scales="free_y")
        )
        splot.save(
            score_dir / f"{concept}.svg", **plot_params("sae_concept_scores", CONFIG)
        )


# * Entry point

if fn := globals().get(smk.rule):
    fn()
elif smk.rule.startswith("compare_"):
    embedding_comparison()
