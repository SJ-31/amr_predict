from pathlib import Path


include: "shared.smk"


if TEST:
    config["bakta"] = f"{TEST_DATA}/bakta"
    config["genomes"] = f"{TEST_DATA}/genomes"
    config["seq_metadata"] = {
        "hamronization": f"{TEST_DATA}/hamronization_combined_report.tsv"
        "combgc": f"{TEST_DATA}/combgc",
        "ampcombi": f"{TEST_DATA}/Ampcombi_summary_cluster.tsv",
        "bakta": f"{TEST_DATA}/bakta"
    }
    config["sample_metadata"]["file"] = f"{config["data"]["meta"]}/jia_samples.tsv"
    config["sample_metadata"]["id_col"] = "Accession"

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
        embedded=[f"{E_OUTDIR}/{d}" for d in PREPROCESSING.keys()],


rule get_seq_metadata:
    output:
        f"{PROCESSED}/{DATE}/seq_metadata.csv",
    script:
        "scripts/prepare_data.py"


rule make_text_datasets:
    output:
        [directory(f"{S_OUTDIR}/{d}") for d in PREPROCESSING.keys()],
    input:
        rules.get_seq_metadata.output,
    params:
        outdir=S_OUTDIR,
        preprocessing=PREPROCESSING,
    script:
        "scripts/prepare_data.py"


rule make_embedded_datasets:
    input:
        rules.make_text_datasets.output,
    params:
        outdir=E_OUTDIR,
    output:
        [directory(f"{E_OUTDIR}/{d}") for d in PREPROCESSING.keys()],
    script:
        "scripts/prepare_data.py"
