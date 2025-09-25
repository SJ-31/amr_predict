include: "shared.smk"


if TEST:
    pass

OUTDIR = f"{OUT}/embedding_comparison/{DATE}"

in_date = config["in_date"]


rule all:
    input
