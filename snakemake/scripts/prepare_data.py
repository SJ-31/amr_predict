#!/usr/bin/env ipython

import os
from itertools import batched
from pathlib import Path
from typing import Literal

import plotnine as gg
import polars as pl
import torch
import torch.utils.data as td
from amr_predict.profiling import memray_from_smk
from amr_predict.utils import EmbeddingCache, load_as, read_tabular
from datasets import Dataset
from loguru import logger
from scipy import stats
from snakemake.script import snakemake as smk

os.environ["HF_HOME"] = smk.config["huggingface"]

from amr_predict.pooling import (
    LEARNING_POOLING_METHODS,
    STATIC_POOLING_METHODS,
    StaticPooler,
)
from amr_predict.preprocessing import EMBEDDING_METHODS, SeqDataset, SeqEmbedder

CONFIG: dict = smk.config
RCONFIG: dict = smk.config.get(smk.rule)
EMBEDDING: EMBEDDING_METHODS = CONFIG["embedding"]
TEXT_KEY = "sequence_aa" if EMBEDDING == "esm" else "sequence"

logger.enable("amr_predict")
logger.add(sink=smk.log["log"])


# * Utilities


def get_seq_level(
    text_dset_path,
    cache: EmbeddingCache,
    add_metadata=False,
) -> td.Dataset:
    cols = ["sample", TEXT_KEY] if not add_metadata else None
    df = load_as(text_dset_path, "polars", cols)
    dset: td.Dataset = cache.to_dataset(df=df, key_col=TEXT_KEY, new_col="embedding")
    return dset


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
    return pl.concat(dfs, how="diagonal_relaxed")


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
    # Maps of sample ids to their indices in the dataset
    to_compute = batched(
        o_cols[sample_key].sample(n, shuffle=True, with_replacement=replacement), 2
    )
    # Random pairs of samples
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
    hamr = pl.read_csv(
        file, separator="\t", raise_if_empty=False, infer_schema_length=None
    )
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
    return pl.concat(dfs, how="diagonal_relaxed")


# * Rules


# ** Generate metadata
def get_seq_metadata():
    dfs = []
    seq_meta = CONFIG["seq_metadata"]
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
        df: pl.DataFrame = pl.concat(dfs, how="diagonal_relaxed")
        # The ideal way to merge these would be with a join, but is not possible due to
        # the differing start, stop indices
        df.write_csv(smk.output[0])
    else:
        pl.DataFrame().write_csv(smk.output[0])


def make_baseline():
    # Deterministic embedding methods that can process whole genomes
    # - kmer
    # - feature_presence
    savepath: Path = Path(smk.output[0])
    name = savepath.stem
    kws = smk.params["preprocessing"][name]
    method = kws.pop("method", None)
    logger.info(f"Begin processing data with {method}")
    if method == "kmer":
        kws.update(
            {
                "fastas": Path(CONFIG["genomes"]),
                "id_col": CONFIG["sample_metadata"]["id_col"],
                "id_rename": "sample",
            }
        )
    elif method == "feature_presence":
        feature_cols = []
        anno = None
        if (source := kws.pop("source", "bakta")) == "bakta":
            anno = Path(CONFIG["seq_metadata"]["bakta"])
            feature_cols = ["Gene", "Product"]
            kws["id_regexp"] = "(.*)_bakta"
        elif source == "hamr":
            anno = Path(smk.input[0])
            feature_cols = ["hamr_gene_symbol", "hamr_drug_class"]
        kws.update(
            {
                "fasta_annotations": anno,
                "feature_cols": feature_cols,
                "id_col": "sample",
                "read_kws": {"comment_prefix": "# "},
                "metadata_pattern": "*_bakta.tsv",
            }
        )
    sem = SeqEmbedder(method=method, **kws)
    dataset = sem(None)
    dataset.save_to_disk(dataset_path=savepath)
    logger.success(f"Finished proccessing with {method}")


def make_text_dataset():
    embedding_method: EMBEDDING_METHODS = CONFIG["embedding"]
    max_length = CONFIG["embedding_method2max_length"][embedding_method]
    savepath: Path = Path(smk.output[0])
    name = savepath.stem
    kws = smk.params["preprocessing"][name]
    if (remap_file := CONFIG.get("fasta_remap")) and Path(remap_file).exists():
        id_remap = dict(read_tabular(remap_file).iter_rows())
        kws["id_remap"] = id_remap
    if kws["split_method"] == "bakta":
        anno = Path(CONFIG["seq_metadata"]["bakta"])
    else:
        anno = None
    SeqDataset.save_from_fastas(
        fastas=CONFIG["genomes"],
        savepath=savepath,
        id_col=CONFIG["sample_metadata"]["id_col"],
        annotations=anno,
        seq_metadata=smk.input[0],
        max_length=max_length,
        **kws,
    )


def make_embedded_dataset():
    params: dict = CONFIG["embedding_methods"][EMBEDDING]
    kws: dict = params.copy()
    inpath = Path(smk.input[0])
    cache_name = f"{inpath.stem}-{EMBEDDING}_cache"
    savepath = Path(smk.output[0])
    workdir = Path(f"{smk.params['outdir']}") / cache_name
    workdir.mkdir(exist_ok=True)
    if EMBEDDING == "Evo2":
        kws["runscript"] = CONFIG["evo2_runscript"]
    elif EMBEDDING == "seqLens" or EMBEDDING == "esm":
        kws["huggingface"] = CONFIG["huggingface"]
    if EMBEDDING == "esm":
        kws["save_dset"] = inpath
    if not savepath.exists():
        logger.info(f"Embedding dataset `{inpath.stem}` started")
        dset = SeqDataset(
            inpath,
            embedder=SeqEmbedder(
                method=EMBEDDING,
                workdir=workdir,
                only_cache=True,  # WARNING: [2025-12-16 Tue] cache.to_dataset is
                # highly memory-intensive, so don't use it
                with_tokens=CONFIG["save_tokens"],
                token_prop=CONFIG["token_prop"],
                text_key="sequence",  # ESM will add 'sequence_aa'
                **kws,
            ),
        )
        dset.embed(None)
        savepath.write_text("TRUE")
        logger.success(f"Embedding dataset `{inpath.stem}` complete")


def pool_embeddings():
    embedding_ds = smk.input[0]
    inpath = Path(embedding_ds).with_suffix("")
    cache = EmbeddingCache(dir=inpath)
    logger.info("Size of cache {}", len(cache))
    ds_name = inpath.stem.removesuffix(f"-{EMBEDDING}_cache")
    texts_path = Path(smk.params["textdir"]).joinpath(ds_name)

    pname: str = smk.params["pooling"]
    spec = RCONFIG["methods"][pname] or {}

    method: STATIC_POOLING_METHODS | LEARNING_POOLING_METHODS = spec.pop(
        "method", pname
    )
    pooling_kws = {k: v for k, v in RCONFIG.items() if k != "methods"}
    pooling_kws.update(spec)
    savepath = Path(smk.params["outdir"]) / f"{ds_name}-{EMBEDDING}-{pname}"
    figpath = smk.params["plotdir"] / f"{ds_name}-{EMBEDDING}-{pname}.png"
    fig_datapath = smk.params["plotdir"] / f"{ds_name}-{EMBEDDING}-{pname}.csv"
    if not savepath.exists():
        logger.info(f"{inpath.stem} `{method}` pooling: started")
        sp: StaticPooler = StaticPooler(
            method=method,
            # NOTE: we merge with metadata during evals to save space
            sample_metadata=None,
            sample_metadata_key=None,
            **pooling_kws,
        )
        add_metadata = method == "seq_subset"
        dset = get_seq_level(texts_path, cache, add_metadata)
        pooled = sp(dset)
        if pooled[RCONFIG["key"]][:].isnan().all():
            raise ValueError("All of the pooled embeddings are nan...")
        logger.success(f"{inpath.stem} `{method}` pooling: complete")
        pooled.save_to_disk(dataset_path=savepath)
        logger.info(f"{inpath.stem} `{method}` pooling: saved to disk")
    else:
        pooled = load_as(savepath)
    original = get_seq_level(texts_path, cache)
    comparison: pl.DataFrame = compare_pooled(
        original,
        pooled,
        o_key=RCONFIG["embedding_key"],
        p_key=RCONFIG["key"],
        sample_key=RCONFIG["sample_key"],
    )
    corr = stats.spearmanr(comparison["d_original"], comparison["d_pooled"])
    comparison.write_csv(fig_datapath)
    plot = (
        gg.ggplot(comparison, gg.aes(x="d_original", y="d_pooled"))
        + gg.geom_point()
        + gg.ggtitle(
            title="Correlation between sequence & genome embedding distances",
            subtitle=f"Spearman rho = {round(corr.statistic, 2)}, p-value: {round(corr.pvalue, 2)}",
        )
        + gg.xlab("Contig distance")
        + gg.ylab("Genome distance")
    )
    plot.save(filename=figpath, verbose=False, width=15, height=12)


# * Run

memray_from_smk(CONFIG, locals()[smk.rule], smk.log["profile"])
