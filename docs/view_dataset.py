import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import polars as pl
    from pyhere import here

    META = here("data", "meta")
    return META, here, pl


@app.cell
def _(META, here, pl):
    # with open(here(META, "ast_download_previews.json"), "r") as f:
    #     ast_downloads = json.load(f)
    bprojects = pl.read_csv(
        here(META, "ncbi_datasets_bioprojects.tsv"), separator="\t"
    ).filter(pl.col("lookup_success"))
    taxids = pl.read_csv(here(META, "ncbi_datasets_lookup.tsv"), separator="\t").filter(
        pl.col("lookup_success")
    )
    subsampled_file = here("data", "meta", "ast_subsampled.tsv")
    tb: pl.DataFrame = pl.read_csv(subsampled_file, separator="\t")
    all_accessions = bprojects["Assembly BioSample Accession"].unique()
    with_ast_data: set = set(
        pl.read_csv(
            here("data", "meta", "ncbi_all_samples.tsv"),
            separator="\t",
            infer_schema_length=None,
        )["BioSample"]
    )
    bprojects = bprojects.filter(
        pl.col("Assembly BioSample Accession").is_in(with_ast_data)
    )
    bprojects.select(["Assembly Accession", "Assembly BioSample Accession"]).rename(
        {
            "Assembly BioSample Accession": "BioSample",
            "Assembly Accession": "Genome",
        }
    ).unique().write_csv(here(META, "genome2biosample.csv"))
    tb = (
        tb.join(
            bprojects,
            how="left",
            left_on="BioSample",
            right_on="Assembly BioSample Accession",
        )
        .unique("BioSample")
        .drop("BioProject_right")
    )
    return bprojects, taxids, tb


@app.cell
def _(mo):
    mo.md(r"""# Samples with available genomes""")
    return


@app.cell
def _(bprojects):
    bprojects
    return


@app.cell
def _(mo):
    mo.md("## Filter appropriate genomes to download")
    return


@app.cell
def _(here, pl, tb: "pl.DataFrame"):
    filtered = tb.filter(
        (pl.col("Assembly Accession").is_not_null())
        & (pl.col("ANI Check status") == "OK")
    )
    n_bsample = len(set(filtered["BioSample"]))
    n_acc = len(set(filtered["Assembly Accession"]))
    unique_equal = n_bsample == n_acc
    print(f"n. unique genome accessions == n. unique biosample accs {unique_equal}")
    if not unique_equal:
        print(f"{n_bsample=}")
        print(f"{n_acc=}")
    to_download_file = here("data", "temp", "ast_to_download.txt")
    to_download_file.write_text("\n".join(set(filtered["Assembly Accession"])))
    filtered
    return (filtered,)


@app.cell
def _(filtered, here, pl, tb: "pl.DataFrame"):
    to_assemble = tb.filter(~pl.col("BioSample").is_in(filtered["BioSample"]))
    to_assemble_file = here("data", "temp", "ast_to_assemble.txt")
    to_assemble_file.write_text("\n".join(to_assemble["BioSample"]))
    return


@app.cell
def _(mo):
    mo.md(r"""# Candidates for reference genomes""")
    return


@app.cell
def _(taxids):
    taxids
    return


if __name__ == "__main__":
    app.run()
