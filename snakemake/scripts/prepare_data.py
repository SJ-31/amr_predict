#!/usr/bin/env ipython

import os
from itertools import batched
from pathlib import Path
from typing import Literal

import plotnine as gg
import polars as pl
import torch
from amr_predict.utils import discretize_resistance, load_as
from datasets import Dataset
from scipy import stats
from snakemake.script import snakemake as smk

os.environ["HF_HOME"] = smk.config["huggingface"]

from amr_predict.pooling import StaticPooler
from amr_predict.preprocessing import SeqDataset, SeqEmbedder

CONFIG: dict = smk.config
RCONFIG: dict = smk.config.get(smk.rule)

# * Utilities


def format_combgc(
    path: Path,
    id_col: str = "sample",
    seqid_col: str = "seqid",
    start_col: str = "start",
    stop_col: str = "stop",
    prefix: str = "combgc",
) -> pl.DataFrame:
    dfs = []
    to_rename = {
        "sample_id": id_col,
        "contig_id": seqid_col,
        "BGC_start": start_col,
        "BGC_end": stop_col,
    }
    wanted_cols = ["PFAM_domains", "Product_class", "MIBiG_ID", "InterPro_ID"]
    to_rename.update({w: f"{prefix}_{w}" for w in wanted_cols})
    for dir in path.iterdir():
        summary_file = dir / "combgc_summary.tsv"
        if not summary_file.exists():
            continue
        df = pl.read_csv(summary_file, separator="\t", raise_if_empty=False)
        if not df.is_empty():
            dfs.append(
                df.rename(to_rename)
                .select(to_rename.values())
                .with_columns(pl.lit(True).alias("is_bgc"))
            )
    return pl.concat(dfs)


def format_ampcombi(
    file,
    id_col: str = "sample",
    seqid_col: str = "seqid",
    start_col: str = "start",
    stop_col: str = "stop",
    prefix: str = "ampcombi",
) -> pl.DataFrame:
    wanted_cols = ["transporter_protein"]
    to_rename = {
        "sample_id": id_col,
        "CDS_start": start_col,
        "CDS_end": stop_col,
        "contig_id": seqid_col,
    }
    to_rename.update({w: f"{prefix}_{w}" for w in wanted_cols})
    ampcombi = pl.read_csv(
        file, separator="\t", infer_schema_length=None, raise_if_empty=False
    )
    if ampcombi.is_empty():
        return ampcombi
    else:
        return (
            ampcombi.rename(to_rename)
            .select(to_rename.values())
            .with_columns(pl.lit(True).alias("is_amp"))
        )


def compare_pooled(
    original: Dataset,
    pooled: Dataset,
    o_key: str = "embedding",
    p_key: str = "x",
    sample_key: str = "sample",
    sample_key_p: str | None = None,
    n: int | None = None,
    metric: Literal["euclidean", "manhattan"] = "manhattan",
):
    if n is None:
        n = pooled.shape[0]
    n *= 2
    replacement = True if n > pooled.shape[0] else False
    sample_key_p = sample_key_p or sample_key
    p = 1 if metric == "manhattan" else 2
    result = {"x": [], "y": [], "d_original": [], "d_pooled": []}

    o_cols: pl.DataFrame = original.remove_columns(o_key).to_polars()
    p_cols: pl.DataFrame = pooled.remove_columns(p_key).to_polars()
    unique_samples = p_cols[sample_key_p].unique()
    o_lookup = {s: (o_cols[sample_key] == s).arg_true() for s in unique_samples}
    p_lookup = {s: (p_cols[sample_key_p] == s).arg_true() for s in unique_samples}
    to_compute = batched(
        o_cols[sample_key].sample(n, shuffle=True, with_replacement=replacement), 2
    )
    for x, y in to_compute:
        for dset, lookup, key, val_key in zip(
            [original, pooled],
            [o_lookup, p_lookup],
            [o_key, p_key],
            ["d_original", "d_pooled"],
        ):
            x_embed = dset.select(lookup[x])[key][:]
            y_embed = dset.select(lookup[y])[key][:]
            dist = torch.cdist(x_embed, y_embed, p=p)
            if val_key == "d_original":
                dist = dist.mean()
            else:
                dist = dist[0][0]
            result[val_key].append(dist)
        result["x"].append(x)
        result["y"].append(y)
    return pl.DataFrame(result)


def format_hamronization(
    file,
    id_col: str = "sample",
    seqid_col: str = "seqid",
    start_col: str = "start",
    stop_col: str = "stop",
    prefix: str = "hamr",
) -> pl.DataFrame:
    wanted_cols = [
        "gene_symbol",
        "resistance_mechanism",
        "predicted_phenotype",
        "antimicrobial_agent",
        "drug_class",
    ]
    to_rename = {
        "input_file_name": id_col,
        "input_gene_start": start_col,
        "input_gene_stop": stop_col,
        "input_sequence_id": seqid_col,
    }
    to_rename.update({w: f"{prefix}_{w}" for w in wanted_cols})
    hamr = pl.read_csv(file, separator="\t", raise_if_empty=False)
    if hamr.is_empty():
        return hamr
    else:
        hamr = hamr.rename(to_rename)
    to_remove = [
        "\\.mapping.*\\.deeparg",
        "\\.tsv\\.amrfinderplus",
        "\\.txt\\.rgi",
        "_retrieved-genes-.*",
    ]
    for pat in to_remove:
        hamr = hamr.with_columns(pl.col(id_col).str.replace(pat, value=""))
    # Fix weird formatting
    hamr = hamr.with_columns(pl.col(seqid_col).str.replace("_seq.*", value=""))
    hamr = hamr.with_columns(
        pl.struct(id_col, seqid_col)
        .map_elements(
            lambda x: x[seqid_col].replace(f"{x[id_col]}_", ""), return_dtype=pl.String
        )
        .alias(seqid_col)
    )
    as_first = [
        start_col,
        stop_col,
    ] + [f"{prefix}_{w}" for w in wanted_cols]
    return hamr.group_by([id_col, seqid_col]).agg(pl.col(f).first() for f in as_first)


def format_bakta(
    bakta_dir: Path,
    id_col: str = "sample",
    seqid_col: str = "seqid",
    start_col: str = "start",
    stop_col: str = "stop",
    prefix="bakta",
) -> pl.DataFrame:
    dfs = []
    for file in bakta_dir.glob("*_bakta.tsv"):
        sample = file.stem.replace("_bakta.tsv", "")
        rename = {
            "#Sequence Id": seqid_col,
            "Gene": f"{prefix}_gene",
            "Start": start_col,
            "Stop": stop_col,
            "Locus Tag": f"{prefix}_locus_tag",
            "Product": f"{prefix}_product",
            "Type": f"{prefix}_type",
        }
        df = pl.read_csv(
            file,
            separator="\t",
            skip_rows=5,
            infer_schema_length=None,
            raise_if_empty=False,
        )
        if df.is_empty():
            continue
        df = (
            df.rename(rename)
            .select(rename.values())
            .with_columns(pl.lit(sample).alias(id_col))
        )
        dfs.append(df)
    return pl.concat(dfs)


# * Generate metadata
if smk.rule == "get_seq_metadata":
    dfs = []
    seq_meta = smk.config["seq_metadata"]
    rename_kws = seq_meta["renaming_rules"]
    if seq_meta.get("hamronization"):
        dfs.append(format_hamronization(seq_meta["hamronization"], **rename_kws))
    if seq_meta.get("bakta"):
        dfs.append(format_bakta(Path(seq_meta["bakta"]), **rename_kws))
    if seq_meta.get("combgc"):
        dfs.append(format_combgc(Path(seq_meta["combgc"]), **rename_kws))
    if seq_meta.get("ampcombi"):
        dfs.append(format_ampcombi(seq_meta["ampcombi"], **rename_kws))
    if dfs:
        df: pl.DataFrame = pl.concat(dfs, how="diagonal")
        df.write_csv(smk.output[0])
    else:
        pl.DataFrame().write_csv(smk.output[0])

# * Make text datasets
elif smk.rule == "make_text_datasets":
    for name, kwargs in smk.params["preprocessing"].items():
        savepath = Path(f"{smk.params['outdir']}/{name}")
        if kwargs["split_method"] == "bakta":
            anno = Path(CONFIG["seq_metadata"]["bakta"])
        else:
            anno = None
        SeqDataset.save_from_fastas(
            fastas=CONFIG["genomes"],
            savepath=savepath,
            id_col=smk.config["sample_metadata"]["id_col"],
            annotations=anno,
            seq_metadata=smk.input[0],
            **kwargs,
        )
# * Embed
elif smk.rule == "make_embedded_datasets":
    for seq_ds in smk.input:
        inpath = Path(seq_ds)
        savepath = Path(smk.params["outdir"]) / inpath.stem
        if not savepath.exists():
            dset = SeqDataset(
                inpath,
                embedder=SeqEmbedder(
                    huggingface=CONFIG["huggingface"],
                    text_key="sequence",
                    pooling=CONFIG["embedding"].get("pooling", "mean"),
                ),
            )
            dset.embed(savepath)
# * Pool
elif smk.rule == "pool_embeddings":
    methods = RCONFIG.pop("methods")
    discretization = RCONFIG.pop("discretize")
    for embedding_ds in smk.input:
        inpath = Path(embedding_ds)
        for method in methods:
            savepath = Path(smk.params["outdir"]) / f"{inpath.stem}-{method}"
            figpath = Path(smk.params["plotdir"]) / f"{inpath.stem}-{method}.png"
            if not savepath.exists():
                sp: StaticPooler = StaticPooler(
                    method=method,
                    sample_metadata=smk.config["sample_metadata"]["file"],
                    sample_metadata_key=smk.config["sample_metadata"]["id_col"],
                    **RCONFIG,
                )
                pooled = sp(inpath)
                pooled = discretize_resistance(pooled, **discretization)
                pooled.save_to_disk(dataset_path=savepath)
            else:
                pooled = load_as(savepath)
            original = load_as(inpath)
            comparison = compare_pooled(
                original,
                pooled,
                o_key=RCONFIG["embedding_key"],
                p_key=RCONFIG["key"],
                sample_key=RCONFIG["sample_key"],
            )
            corr = stats.spearmanr(comparison["d_original"], comparison["d_pooled"])
            plot = (
                gg.ggplot(comparison, gg.aes(x="d_original", y="d_pooled"))
                + gg.geom_point()
                + gg.ggtitle(
                    title="Spearman correlation between distances of contig embeddings & genome embeddings",
                    subtitle=f"rho = {round(corr.statistic, 2)}, p-value: {round(corr.pvalue, 2)}",
                )
                + gg.xlab("Contig distance")
                + gg.ylab("Genome distance")
            )
            plot.save(filename=figpath, verbose=False)
