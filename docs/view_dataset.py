import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    from pyhere import here

    META = here("data", "meta")
    return META, cs, here, mo, pl


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
    mo.md("""## Filter appropriate genomes to download""")
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
    to_assemble_file.write_text("\n".join(to_assemble["Run"]))
    return (to_assemble,)


@app.cell
def _(mo):
    mo.md(r"""# Candidates for reference genomes""")
    return


@app.cell(hide_code=True)
def _(cs, pl, taxids, to_assemble):
    required_id = taxids.filter(
        pl.col("taxid_query").is_in(to_assemble["TaxID"])
    ).filter(pl.col("ANI Check status") == "OK")
    ani_status_scoring = {
        "species_match": 1.2,
        "synonym_match": 1.2,
        "derived_species_match": 1.5,  # At the subspecies level,
        # but could a different subspecies to the submitted
        "subspecies_match": 2.0,
        "below_threshold_mismatch": 1.0,
    }
    level_scoring = {
        "Chromosome": 1.2,
        "Complete Genome": 1.3,
        "Scaffold": 1.1,
        "Contig": 1.0,
    }
    required_id = (
        required_id.with_columns(
            pl.col("ANI Best match status")
            .replace(ani_status_scoring)
            .alias("ani_match_score"),
            pl.col("Assembly Refseq Category")
            .replace({None: 1.0, "reference genome": 1.5})
            .alias("refseq_category_score"),
            pl.col("Assembly Level")
            .replace(level_scoring)
            .alias("assembly_level_score"),
        )
        .cast({cs.ends_with("_score"): pl.Float64})
        .with_columns(
            (
                pl.col("ani_match_score")
                * pl.col("refseq_category_score")
                * pl.col("assembly_level_score")
                * pl.col("ANI Best ANI match ANI")
            ).alias("score")
        )
    )
    required_id
    return (required_id,)


@app.cell
def _(pl, required_id):
    dfs = []
    per_group = 3
    for taxid, group in required_id.group_by("taxid_query"):
        print(f"{taxid=}")
        group = group.filter(pl.col("Assembly Atypical Is Atypical").is_null()).unique(
            "Assembly Accession"
        )
        print(group.shape)
        nrow, _ = group.shape
        if nrow <= per_group:
            dfs.append(group)
            continue
        top = group.top_k(per_group, by="score")
        print(top)
        dfs.append(top)
    return (dfs,)


@app.cell
def _(META, dfs, here, pl):
    ref_df = pl.concat(dfs)
    ref_df.write_csv(here(META, "reference_genomes.tsv"), separator="\t")
    here("data", "temp", "reference_accs.txt").write_text(
        "\n".join(ref_df["Assembly Accession"])
    )
    return


if __name__ == "__main__":
    app.run()
