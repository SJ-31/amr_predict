#!/usr/bin/env ipython

from pathlib import Path

import polars as pl
from amr_predict.utils import SeqDataset, SeqEmbedder
from snakemake.script import snakemake as smk

TEST: bool = smk.config["test"]
CONFIG: dict = smk.config


# * Generate metadata
if smk.rule == "generate_metadata":
    if TEST:
        file = f"{smk.config["data"]["meta"]}/jia_samples.tsv"
        df = pl.read_csv(file, separator="\t", infer_schema_length=None).rename(
            {"Accession": "sample"}
        )
        df.write_csv(smk.output[0])
    else:
        raise NotImplementedError("")
        # TODO: should wait until funcscan finishes to see what you can do here
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
            metadata=smk.input["meta"][0],
            savepath=savepath,
            id_col="sample",
            annotations=anno,
            **kwargs,
        )
# * Embed
elif smk.rule == "embed":
    for seq_ds in smk.input:
        inpath = Path(seq_ds)
        savepath = Path(smk.params["outdir"]).joinpath(inpath.stem)
        dset = SeqDataset(
            inpath,
            embedder=SeqEmbedder(
                huggingface=CONFIG["huggingface"],
                text_key="sequence",
                pooling=CONFIG["embedding"].get("pooling", "mean"),
            ),
        )
        dset.embed(savepath)
