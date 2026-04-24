from pathlib import Path


configfile: "data_sources.yaml"


include: "Snakefile"


assembly_outdir = f"{config["remote"]}/downloaded_assemblies"
fastq_outdir = f"{config["remote"]}/fastq"


rule all:
    input:
        f"{TEMP}/biosample_mappings.tsv",
        f"{TEMP}/bioproject_samples.tsv",


rule get_bioproject:
    output:
        f"{TEMP}/bioproject_samples.tsv",
    run:
        import pandas as pd

        dfs = []
        for f in config["bioproject_files"]:
            cur = (
                pd.read_csv(
                    f"{META}/{f}",
                    sep="\t",
                    skiprows=1,
                    header=0,
                    names=("Level", "WGS", "BioSample", "Strain", "Taxonomy", "_"),
                )
                .drop("_", axis=1)
                .reset_index(names="Assembly")  # Required due to weird file formatting
            )
            dfs.append(cur)
        samples = pd.concat(dfs)
        samples.to_csv(output[0], sep="\t", index=False)


rule get_biosample_mapping:
    output:
        out=f"{TEMP}/biosample_mappings.tsv",
        # tmp=temp(f"{TEMP}/tmp_mappings.tsv"),
        acc=temp(f"{TEMP}/biosample_accs.txt"),
    input:
        f"{TEMP}/bioproject_samples.tsv",
    params:
        cache=f"{config['cache']}/rentrez",
    run:
        import pandas as pd

        together = set()
        for sheet, col in config["biosample_sheets"].items():
            if ".tsv" in sheet:
                df = pd.read_csv(f"{RAW}/{sheet}", sep="\t")
            else:
                df = pd.read_csv(f"{RAW}/{sheet}")
            accs = set([str(s) for s in set(df[col])])
            together |= accs
        from_bioproject = pd.read_csv(input[0], sep="\t")
        together |= set(from_bioproject["BioSample"])
        with open(output.acc, "w") as f:
            f.write("\n".join(together))

        shell(
            f"Rscript {SRC}/R/map_biosample.R -i {output.acc} -o {output.out} -c {params.cache}"
        )


# rule get_raw:
#     input:
#         f"{TEMP}/biosample_mappings.tsv",
#     output:
#         scc,
#     run:
#         import os
#         from subprocess import run
#         df = pd.read_csv(input[0], sep="\t")
#         tracker = {"Run": [], "BioSample": [], "Success": []}
#         for bsample, run in zip(df["BioSample"], df["Run"]):
#             cur = Path(f"{fastq_outdir}/{bsample}")
#             if not cur.exists():
#                 run(f"prefetch {run} -O {cur}", shell=True)
#                 run(f"fasterq ./{run}", shell=True)
