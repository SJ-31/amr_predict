configfile: "shared.yaml"


include: "shared.smk"


ast_browser = f"{RAW}/asts.tsv"
bvbrc_genome = f"{RAW}/BVBRC_genome.csv"


rule all:
    input:
        f"{TEMP}/biosample_mappings.tsv",


rule get_biosample_mapping:
    output:
        out=f"{TEMP}/biosample_mappings.tsv",
        tmp=temp(f"{TEMP}/tmp_mappings.tsv"),
        acc=temp(f"{TEMP}/biosample_accs.txt"),
    run:
        import pandas as pd

        ast_df = pd.read_csv(ast_browser, sep="\t")
        bvbrc_df = pd.read_csv(bvbrc_genome)
        ast_accs, bvbrc_accs = set(ast_df["#BioSample"]), set(
            bvbrc_df["BioSample Accession"]
        )
        with open(output.acc, "w") as f:
            f.write("\n".join(ast_ccs | bvbrc_accs))
        shell(f"Rscript {SRC}/R/map_biosample.R -i {output.acc} -o {output.tmp}")
        result = pd.read_csv(output.tmp, sep="\t")
        source_df = (
            ast_df.loc[:, ["#BioSample"]]
            .drop_duplicates()
            .assign(A="AST_Browser")
            .merge(
                bvbrc_df.loc[:, ["BioSample Accession"]]
                .drop_duplicates()
                .assign(B="BVBRC"),
                left_on="#BioSample",
                right_on="BioSample Accession",
            )
        )
        source_df.loc[:, "Source"] = source_df["A"].combine_first(source_df["B"])
        source_df = source_df.drop(["A", "B"], axis=1)
        result = result.merge(
            source_df, left_on="#BioSample", right_on="BioSample", how="left"
        )
        result.to_csv(output.out, sep="\t")
