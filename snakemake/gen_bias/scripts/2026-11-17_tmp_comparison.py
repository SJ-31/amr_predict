#!/usr/bin/env ipython

import itertools
from collections import defaultdict

# [2026-11-17 Tue] TODO: move this into the main script when finished
from importlib import util
from math import log
from typing import Literal

import networkx as nx
import numpy as np
import plotnine as gg
import polars as pl
import scipy.spatial.distance as spd
import scipy.stats as stats
import yte
from pyhere import here

gdir = here("data", "remote", "2026-09-16_gen_bias")


def load_file_as_module(name, location):
    spec = util.spec_from_file_location(name, location)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# * Sorting out the comparisons


mod = load_file_as_module("snakemake", here("snakemake", "gen_bias", "env.py"))

with open(here("snakemake", "gen_bias", "all_models.yaml")) as f:
    model_relations = mod.model_relations(yte.process_yaml(f)["models"])

# Example for gathering files to compare effect of hyperparams
prompt = "e_coli_16s_no_v"
model = "evo-1-131k-base"
param = "temperature"
gendir = here(gdir, "generated")
files_read = {
    k: gendir / v / f"{prompt}.csv"
    for k, v in model_relations["model_groups"][model][param].items()
}

# Example for gathering files to compare effect of model across hyperparams
temp_value = "0.4"
files_read = {
    k: gendir / v / f"{prompt}.csv"
    for k, v in model_relations["param_groups"][param][temp_value].items()
}

# Example of gathering files to compare prompts
prompts_compare = [  # TODO: make this a parameter in env yaml
    prompt,
    "e_coli_16s_v1_v3",
    "e_coli_16s_v1_v6",
    "d_yakuba_coi_forward",
]
files_read = {
    p: gendir / model_relations["model_groups"][model][param][temp_value] / f"{p}.csv"
    for p in prompts_compare
}


def order_comparisons(
    df: pl.DataFrame,
    two_sided_col: str,
    one_sided_col: str,
    x_col: str,
    y_col: str,
    threshold: float = 0.05,
    alternative: Literal["greater", "less"] = "greater",
) -> dict[str, float]:
    """
    Parse a series of pairwise hypothesis tests to determine which group is the
    largest or smallest

    Parameters
    ----------
    two_sided_col : str
        Column in `df` containing p-values for a two-sided test
    one_sided_col : str
        Column in `df` containing p-values for a one-sided test
    alternative : str
        Alternative hypothesis used in the one-sided test
        'greater' is understood as 'x greater', likewise for 'less'

    Returns
    -------
    Dictionary mapping group names to their ranks.
    Groups with an ambiguous relationship (i.e. which form a cycle) are given
    a rank of -1
    Groups with NA values are given a rank of NA
    """
    G: nx.DiGraph = nx.DiGraph()
    no_differences = {}
    ambiguous = set()
    for row in df.filter(
        (pl.col(two_sided_col).is_not_null() & pl.col(two_sided_col).is_not_nan())
    ).iter_rows(named=True):
        p_two_sided, p_one_sided = row[two_sided_col], row[one_sided_col]
        x, y = row[x_col], row[y_col]
        if p_two_sided > threshold:
            no_differences[x] = y
            no_differences[y] = x
            continue
        if x not in G:
            G.add_node(x)
        if y not in G:
            G.add_node(y)
        try:
            weight = -log(p_one_sided)
        except ValueError:
            weight = -log(1e-10)
        if (p_one_sided < threshold and alternative == "greater") or (
            p_one_sided > threshold and alternative == "less"
        ):
            G.add_edge(x, y, p=weight)
        else:
            G.add_edge(y, x, p=weight)
    try:
        cycle = nx.find_cycle(G)
        for edge in cycle:
            G.remove_edge(*edge)
            ambiguous |= set(edge)
    except nx.NetworkXNoCycle:
        pass
    path = nx.dag_longest_path(G, weight="p")
    result = {str(n): float(i + 1) for i, n in enumerate(path)}
    original_ranks = result.copy()
    all_nodes = set(df[x_col]) | set(df[y_col])
    left_out = [n for n in all_nodes if n not in path]
    if not left_out:
        return result
    num_matched = {}
    for n in left_out:
        if n in ambiguous:
            result[n] = -1
        elif n in no_differences:
            matching = no_differences[n]
            if matching in original_ranks:
                rank = original_ranks[matching]
                n_prev = num_matched.get(matching, 1)
                result[n] = rank + 1 / (n_prev + 1)
                result[matching] = rank + 1 / (n_prev + 1)
                num_matched[matching] = num_matched.get(matching, 1) + 1
            else:
                result[n] = len(all_nodes)
        else:
            result[n] = np.nan
    return result


# * Protein descriptors


def compare_descriptors(
    dfs: list[pl.DataFrame], names: list | None = None, threshold: float = 0.05
):
    """Compare protein descriptor values within a results set

    Parameters
    ----------
    dfs : list[pl.DataFrame]
        Dataframes containing the raw descriptor values
    names : list
        Annotation for each descriptor set e.g. hyperparameter values for comparing
        the effect of hyperparameters, model names for comparing between different models

    Returns
    -------
    A tuple of two plot objects
    - Density plot showing the distribution of generated values with the value of the prompt
      sequence subtracted. For depicting "novelty"
    - Box plots showing the distribution of generated values. For depicting consistency
      of the generated sequences with respect to each measure
    """
    tmp_novelty: dict[str, pl.DataFrame]
    tmp_gens: dict[str, pl.DataFrame]
    tmp_novelty, tmp_gens = {}, {}
    oris: dict[str, dict] = {}
    if not names:
        names = list(range(len(dfs)))
    # TODO: plot the effect of the hyperparam on the novelty measures
    # Hyperparams are assumed to be numeric, float
    # TODO: add a `hyperparam_blacklist` to SnakeEnv to ignore named hyperparameters
    descriptors = set()
    for df, n in zip(dfs, names):
        descriptors |= set(df.columns) - {"id"}
        orig = df.filter(pl.col("id").str.ends_with("TRIMMED"))
        assert orig.height == 1, (
            "There should only be one measure for the original prompt."
        )
        vals = orig.drop("id").row(0, named=True)
        oris[n] = vals
        df = df.with_columns(pl.lit(n).alias("name"))
        gen = df.filter(~pl.col("id").str.ends_with("TRIMMED"))
        tmp_gens[n] = gen
        tmp_novelty[n] = df.drop("id").with_columns(
            (pl.col(k) - v).abs() for k, v in vals.items()
        )
    # Testing for which value set has the greatest deviation from the prompt sequence,
    # and the probability of observing the prompt sequence in the distribution
    # of the generated sequences
    test_tmp = {
        "x": [],
        "y": [],
        "descriptor": [],
        "p_value": [],
        "statistic": [],
        "p_value_x_greater": [],
        "statistic_x_greater": [],
    }
    for x, y in itertools.combinations(names, 2):
        for desc in descriptors:
            test = stats.ks_2samp(tmp_novelty[x][desc], tmp_novelty[y][desc])
            test_tmp["descriptor"].append(desc)
            test_tmp["x"].append(x)
            test_tmp["y"].append(y)
            test_tmp["p_value"].append(test.pvalue)
            test_tmp["statistic"].append(test.statistic)
            if test.pvalue < threshold:
                test2 = stats.ks_2samp(
                    tmp_novelty[x][desc], tmp_novelty[y][desc], alternative="greater"
                )
                test_tmp["p_value_x_greater"].append(test2.pvalue)
                test_tmp["statistic_x_greater"].append(test2.statistic)
            else:
                test_tmp["p_value_x_greater"].append(None)
                test_tmp["statistic_x_greater"].append(None)
    test_novelty_df = pl.DataFrame(test_tmp).with_columns(
        pl.Series(
            stats.false_discovery_control(test_tmp["p_value"], method="bh")
        ).alias("p_adj"),
        pl.Series(
            stats.false_discovery_control(test_tmp["p_value_x_greater"], method="bh")
        ).alias("p_adj_x_greater"),
    )
    test_pr_tmp = {"name": [], "p": [], "descriptor": []}
    for n, df in tmp_novelty.items():
        for desc in descriptors:
            est = stats.estimated_cdf(tmp_novelty[n][desc], oris[n][desc])
            test_pr_tmp["name"].append(n)
            test_pr_tmp["p"].append(est)
            test_pr_tmp["descriptor"].append(desc)

    tmp_ordering = {"descriptor": [], "name": [], "rank": []}
    for desc in descriptors:
        order = order_comparisons(
            test_novelty_df.filter(pl.col("descriptor") == desc),
            x_col="x",
            y_col="y",
            two_sided_col="p_adj",
            one_sided_col="p_adj_x_greater",
            alternative="greater",
        )
        for k, v in order.items():
            tmp_ordering["descriptor"].append(desc)
            tmp_ordering["name"].append(k)
            tmp_ordering["rank"].append(v)

    novelty_long = (
        pl.concat(tmp_novelty.values())
        .unpivot(index="name")
        .cast({"value": pl.Float64, "name": str})
    )
    # plot = (
    #     gg.ggplot(novelty_long, gg.aes(x="value", color="name"))
    #     + gg.geom_density()
    #     + gg.facet_wrap("variable", scales="free_x")
    # )
    pr_df = pl.DataFrame(test_pr_tmp)
    joined = novelty_long.join(
        pr_df.rename({"descriptor": "variable"}),
        on=["name", "variable"],
    ).join(
        pl.DataFrame(tmp_ordering)
        .rename({"descriptor": "variable"})
        .with_columns(pl.col("rank").cast(pl.String)),
        on=["name", "variable"],
    )
    plot = (
        gg.ggplot(joined, gg.aes(x="name", y="value", fill="rank"))
        + gg.geom_violin(position="identity", style="right")
        + gg.coord_flip()
        + gg.facet_wrap("variable", scales="free_x", nrow=2)
        + gg.theme(
            axis_ticks_minor=gg.element_blank(),
            axis_ticks_length_major=gg.element_blank(),
            figure_size=(12, 12),
        )
    )
    return novelty_long, test_novelty_df, pr_df, joined, plot


descriptor_dir = gdir / "protein_descriptors"

dfs = [
    descriptor_dir / "evo2_t1" / "gsha-raw.csv",
    descriptor_dir / "evo2_t0" / "gsha-raw.csv",
    descriptor_dir / "evo2_t2" / "gsha-raw.csv",
    descriptor_dir / "evo2_t3" / "gsha-raw.csv",
    descriptor_dir / "evo2_t4" / "gsha-raw.csv",
    descriptor_dir / "evo2_t5" / "gsha-raw.csv",
]


names = ["t1", "t0", "t2", "t3", "t4", "t5"]

long = compare_descriptors([pl.read_csv(d) for d in dfs], names)


# color with p-value, and rank by statistical significance


# * Motifs

motif_dir = gdir / "motifs"

sm1 = pl.read_csv(motif_dir / "evo_t3" / "d_yakuba_coi_forward.tsv", separator="\t")
sm2 = pl.read_csv(motif_dir / "evo_t0" / "d_yakuba_coi_forward.tsv", separator="\t")


def motif_stats(
    df: pl.DataFrame, name: str
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    summaries = defaultdict(list)
    df.group_by("sequence_name").agg("motif_id").with_columns(
        pl.col("motif_id").list.len().alias("n"),
        pl.col("motif_id").list.n_unique().alias("n_unique"),
    )
    count_mat = (
        df.select("motif_id", "sequence_name")
        .with_columns(pl.lit(1).alias("count"))
        .pivot("motif_id", values="count", aggregate_function="sum")
    )
    bc_dist_within = spd.pdist(
        count_mat.filter(~pl.col("sequence_name").str.ends_with("-TRIMMED")).drop(
            "sequence_name"
        ),
        "braycurtis",
    )
    orig = count_mat.filter(pl.col("sequence_name").str.ends_with("-TRIMMED"))
    assert orig.height == 0 or orig.height == 1, (
        "There should only be one non-generated sequence in the set"
    )
    if orig.height == 1:
        orig_vec = orig.drop("sequence_name")
    else:
        orig_vec = np.array([0] * count_mat.shape[1])
    bc_dist_to_orig = spd.cdist(count_mat.drop("sequence_name"), orig_vec, "braycurtis")
    for metric, fn in zip(
        ("mean", "median", "std", "range"),
        (
            lambda x: np.mean(x),
            lambda x: np.median(x),
            lambda x: np.std(x),
            lambda x: np.max(x) - np.min(x),
        ),
    ):
        summaries["metric"].extend([metric] * 2)
        summaries["distance_category"].extend(["within", "to_orig"])
        summaries["value"].append(fn(bc_dist_within))
        summaries["value"].append(fn(bc_dist_to_orig))
    distribution_within = pl.DataFrame({"value": bc_dist_within.flatten()})
    distribution_ref_comparison = pl.DataFrame({"value": bc_dist_to_orig[:, 0]})
    return (
        pl.DataFrame(summaries).with_columns(pl.lit(name).alias("name")),
        distribution_within.with_columns(pl.lit(name).alias("name")),
        distribution_ref_comparison.with_columns(pl.lit(name).alias("name")),
    )


# TODO: [2026-11-28 Sat] compute these and store in parquet file for plotting later.
# You can retrieve in R either with arrow or duckdb
