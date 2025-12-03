#!/usr/bin/env ipython

import json
import subprocess as sp
from functools import reduce
from pathlib import Path
from typing import Literal

import polars as pl
from pyhere import here

META: Path = here("data", "meta")

metadata: pl.DataFrame = pl.read_csv(here(META, "ast_subsampled.tsv"), separator="\t")
workdir: Path = here("data", "temp", "ncbi-datasets")
workdir.mkdir(exist_ok=True)

taxids = metadata["TaxID"].unique()

summary_cols = [
    "accession",
    "organism-tax-id",
    "organism-name",
    "assmstats-number-of-contigs",
    "ani-best-match-status",
    "ani-best-ani-match-ani",
    "ani-best-ani-match-category",
    "ani-category",
    "ani-check-status",
    "annotinfo-busco-complete",
    "assminfo-atypicalis-atypical",
    "assminfo-atypicalwarnings",
    "assminfo-level",
    "assminfo-refseq-category",
    "assminfo-release-date",
    "assmstats-contig-l50",
    "assmstats-total-sequence-len",
    "assminfo-biosample-accession",
    "assminfo-bioproject",
]

col_flag = ",".join(summary_cols)


def lookup(
    id, type: Literal["taxon", "accession"] = "taxon", query_col="taxid_query"
) -> pl.DataFrame:
    outfile = workdir.joinpath(f"{id}.tsv")
    if not outfile.exists():
        command = f"""
        datasets summary genome {type} {id} --as-json-lines | dataformat tsv genome --fields {col_flag} > {outfile}
        """
        proc = sp.run(command, shell=True)
        proc.check_returncode()
    try:
        df = pl.read_csv(
            outfile, separator="\t", infer_schema_length=None
        ).with_columns(
            pl.lit(id).alias(query_col), pl.lit(True).alias("lookup_success")
        )
    except pl.exceptions.NoDataError:
        df = pl.DataFrame({query_col: id, "lookup_success": False})
    return df


def download_preview(acc) -> dict:
    command = f"""
    datasets download genome accession {acc} --preview
"""
    outfile = workdir.joinpath(f"preview_{acc}.json")
    if not outfile.exists():
        proc = sp.run(command, shell=True, capture_output=True)
        proc.check_returncode()
        res = json.loads(proc.stdout.decode())
        with open(outfile, "w") as f:
            json.dump(res, f)
    else:
        with open(outfile, "w") as f:
            res = json.load(res)
    return {acc: res}


if __name__ == "__main__":
    dfs = [lookup(t) for t in taxids]
    pl.concat(dfs, how="diagonal_relaxed").write_csv(
        here(META, "ncbi_datasets_lookup.tsv"), separator="\t"
    )
    bioprojects = pl.read_csv(
        here(META, "biosample_mapping_2025-11-21.csv"), infer_schema_length=None
    )["BioProject"].unique()
    bioprojects = list(filter(lambda x: x and x.startswith("PRJ"), bioprojects))
    bproject_dfs = [
        lookup(a, type="accession", query_col="BioProject") for a in bioprojects
    ]
    pl.concat(bproject_dfs, how="diagonal_relaxed").write_csv(
        here(META, "ncbi_datasets_bioprojects.tsv"), separator="\t"
    )
    previews = reduce(
        lambda x, y: {**x, **y}, [download_preview(a) for a in bioprojects]
    )
    with open(here(META, "ast_download_previews.json"), "w") as f:
        json.dump(previews, f)
