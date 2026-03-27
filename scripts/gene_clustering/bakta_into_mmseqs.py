#!/usr/bin/env ipython

from pathlib import Path

import polars as pl
from amr_predict.preprocessing import SeqDataset
from amr_predict.utils import load_as
from pyhere import here

REQUIRED: tuple = ("gene", "product", "type")


remote = here("data", "remote")

mm_out = here(
    "data",
    "remote",
    "bakta_mmseqs_2026-03-25",
)
dset_path = mm_out / "sequence_dataset"


def preprocess_full_len():
    SeqDataset.save_from_fastas(
        split_method="bakta",
        savepath=dset_path,
        max_length=None,
        fastas=remote / "genomes/ast_browser_batches/all",
        annotations=remote / "output/ast_browser/funcscan/all/bakta",
        utr_amount=None,
        upstream_context=0,
        downstream_context=0,
    )


def bakta_into_fasta(
    dset: pl.DataFrame, prefix: str, tax_map: pl.DataFrame, workdir: Path
):
    """
    Fasta header format is
    sample|bakta_gene|bakta_product|bakta_type
    """
    dset = (
        dset.select("sample", "sequence", *REQUIRED)
        .filter(pl.any_horizontal(pl.col(REQUIRED).is_not_null()))
        .with_columns(pl.col(REQUIRED).fill_null(""))
        .with_columns(pl.row_index().alias("mmseqs_acc").cast(pl.String))
        .with_columns(mmseqs_acc="MM" + pl.col("mmseqs_acc"))
    )
    dset = dset.join(tax_map, on="sample", how="left")
    fasta_rep = ">" + dset["mmseqs_acc"] + "\n" + dset["sequence"]
    meta_write = dset.select("mmseqs_acc", "sample", "taxid", *REQUIRED)

    meta_write.write_csv(workdir / f"{prefix}mmseqs_db_metadata.csv")
    with open(workdir / f"{prefix}mmseqs.fasta", "w") as f:
        f.write("\n".join(fasta_rep))


tax_map = (
    pl.read_csv(
        here("data", "meta", "biosample_mapping_2025-11-21.csv"),
        infer_schema_length=None,
    )
    .rename({"Query": "sample", "TaxID": "taxid"})
    .select("sample", "taxid")
    .with_columns(pl.col("taxid").cast(pl.String).fill_null(""))
)


def write_fasta():
    dset = load_as(dset_path, "polars")
    print(dset)
    for req in REQUIRED:
        print(dset[req].value_counts())
    bakta_into_fasta(dset, prefix="2026-03-25_", tax_map=tax_map, workdir=mm_out)


def collect_clusters():
    clustering = pl.read_csv(
        mm_out / "clusters.tsv", new_columns=["rep", "member"], separator="\t"
    ).with_columns(
        pl.len().over("rep").alias("cluster_size"),
        cluster="c"
        + pl.col("rep").rank().alias("cluster").cast(pl.Int64).cast(pl.String),
    )

    meta = pl.read_csv(mm_out / "metadata.csv").join(
        clustering, how="left", left_on="mmseqs_acc", right_on="member"
    )

    meta = meta.with_columns(
        is_rep=pl.col("mmseqs_acc") == pl.col("rep"),
        gene=pl.col("gene").replace("", None),
        product=pl.col("product").replace("", None),
    )
    meta.write_csv(mm_out / "metadata.csv")

    exprs: dict = {
        "group_unique": [pl.col(c).unique() for c in REQUIRED],
        "drop_nulls": [pl.col(c).list.drop_nulls() for c in REQUIRED],
        "count": [pl.col(c).list.len().alias(f"n_{c}") for c in REQUIRED],
        "join": [pl.col(c).list.join("@") for c in REQUIRED],
    }
    cluster_genes = (
        meta.group_by("cluster")
        .agg(*exprs["group_unique"])
        .with_columns(*exprs["drop_nulls"])
        .with_columns(*exprs["count"])
        .with_columns(*exprs["join"])
    )
    cluster_genes.write_csv(mm_out / "clusters_meta.csv")


# preprocess_full_len()  # NOTE: completed [2026-03-27 Fri]

# write_fasta()  # NOTE: completed [2026-03-27 Fri]

# collect_clusters() # TODO: run this after re-creating the db with full-length seqs
