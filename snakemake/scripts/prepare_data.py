#!/usr/bin/env ipython

from pathlib import Path

import polars as pl
from amr_predict.utils import SeqDataset, SeqEmbedder
from snakemake.script import snakemake as smk

TEST: bool = smk.config["test"]
CONFIG: dict = smk.config

# * Utilities


def format_hamronization(
    file,
    id_col: str = "sample",
    seqid_col: str = "seqid",
    start_col: str = "start",
    stop_col: str = "stop",
) -> pl.DataFrame:
    hamr = pl.read_csv(file, separator="\t").rename(
        {
            "input_file_name": id_col,
            "input_gene_start": start_col,
            "input_gene_stop": stop_col,
            "input_sequence_id": seqid_col,
        }
    )
    to_remove = [
        "\\.mapping.*\\.deeparg",
        "\\.tsv\\.amrfinderplus",
        "\\.txt\\.rgi",
        "_retrieved-genes-.*",
    ]
    for pat in to_remove:
        hamr = hamr.with_columns(sample=pl.col("sample").str.replace(pat, value=""))
    as_first = [
        start_col,
        stop_col,
        "gene_symbol",
        "resistance_mechanism",
        "predicted_phenotype",
        "antimicrobial_agent",
        "drug_class",
    ]
    return hamr.group_by([id_col, seqid_col]).agg(pl.col(f).first() for f in as_first)


# * Generate metadata
# TODO: have this rule format the summarized amr prediction files from the funcscan
# This includes
# 1. hamronization DONE
# 2. ampcombi
# 3. combgc
# You want to pass the result as the seq_meta argument of SeqDataset
if smk.rule == "get_seq_metadata":
    dfs = []
    seq_meta = smk.config["seq_metadata"]
    if seq_meta.get("hamronization"):
        dfs.append(format_hamronization(seq_meta["hamronization"]))
        print(dfs)
    elif seq_meta.get("combgc"):
        raise NotImplementedError()
    elif seq_meta.get("ampcombi"):
        raise NotImplementedError()
    # TODO: each file essentially needs to have four cols: sample, seqid, start, stop
    if dfs:
        df: pl.DataFrame = pl.concat(dfs)
        df.write_csv(smk.output[0])
    else:
        pl.DataFrame().write_csv(smk.output[0])

# * Make text datasets
elif smk.rule == "make_text_datasets":
    for name, kwargs in smk.params["preprocessing"].items():
        savepath = Path(f"{smk.params['outdir']}/{name}")
        if kwargs["split_method"] == "bakta":
            anno = Path(CONFIG["bakta"])
        else:
            anno = None
        SeqDataset.save_from_fastas(
            fastas=CONFIG["genomes"],
            metadata=smk.config["sample_metadata"]["file"],
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
        savepath = Path(smk.params["outdir"]).joinpath(inpath.stem)
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
