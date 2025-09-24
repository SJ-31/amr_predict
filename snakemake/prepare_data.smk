from pathlib import Path


configfile: "shared.yaml"


include: "shared.smk"


if TEST:
    config["bakta"] = f"{TEST_DATA}/bakta"
    config["genomes"] = f"{TEST_DATA}/genomes"
    # TODO: configure the annotation sources
    # config["funcscan"]

PREPROCESSING = config["preprocessing"]
S_OUTDIR = (
    f"{REMOTE}/datasets/processed_sequences/{DATE}"  # Datasets containing processed
)
# genome data e.g. split into ORFs, binned etc.
E_OUTDIR = (
    f"{REMOTE}/datasets/embedded/{DATE}"  # Datasets containing embedded, processed
)


rule all:
    input:
        embedded=[directory(f"{E_OUTDIR}/{d}") for d in PREPROCESSING.keys()],


rule generate_metadata:
    output:
        f"{PROCESSED}/{DATE}/metadata.csv",
    script:
        "scripts/prepare_data.py"


rule make_text_datasets:
    output:
        [directory(f"{E_OUTDIR}/{d}") for d in PREPROCESSING.keys()],
    input:
        meta=rules.generate_metadata.output,
    params:
        outdir=S_OUTDIR,
        preprocessing=PREPROCESSING,
        genomes=f"{REMOTE}/{config['genome_dir']}",
    script:
        "scripts/prepare_data.py"


rule embed:
    input:
        rules.make_text_datasets.output,
    params:
        outdir=E_OUTDIR,
    output:
        rules.all.embedded,
    script:
        "scripts/prepare_data.py"
