from pathlib import Path


include: "Snakefile"


if TEST:
    config["bakta"] = f"{TEST_DATA}/bakta"
    config["genomes"] = f"{TEST_DATA}/genomes"
    config["seq_metadata"].update(
        {
            "hamronization": f"{TEST_DATA}/hamronization_combined_report.tsv",
            "combgc": f"{TEST_DATA}/combgc",
            "ampcombi": f"{TEST_DATA}/Ampcombi_summary_cluster.tsv",
            "bakta": f"{TEST_DATA}/bakta",
        }
    )
    config["sample_metadata"]["file"] = f"{config["data"]["meta"]}/jia_samples.tsv"
    config["sample_metadata"]["id_col"] = "Accession"
    config["pooling"].update({"obs_keep": ["AMK", "GEN", "IPM", "CRO"]})

PREPROCESSING = config["preprocessing"]
DATA_OUTS = {
    k: f"{REMOTE}/datasets/{s}/{DATE}"
    for k, s in zip(
        ["S", "E", "P"],
        ["processed_sequences", "embedded", "pooled"],
        # Datasets are...
        # 1. Processed genome data e.g. split into ORFs, binned etc.
        # 2. Sequences embedded by the chosen GLM
        # 3. Sequences pooled into genome-level representations
    )
}


rule all:
    input:
        embedded=[f"{DATA_OUTS["E"]}/{d}" for d in PREPROCESSING.keys()],
        pooled=expand(
            "{o}/{d}-{p}",
            o=DATA_OUTS["P"],
            d=PREPROCESSING.keys(),
            p=config["pooling"]["methods"],
        ),
        meta=f"{PROCESSED}/{DATE}/seq_metadata.csv",


rule get_seq_metadata:
    output:
        rules.all.input.meta,
    script:
        "scripts/prepare_data.py"


rule make_text_datasets:
    output:
        [directory(f"{DATA_OUTS["S"]}/{d}") for d in PREPROCESSING.keys()],
    input:
        rules.get_seq_metadata.output,
    params:
        outdir=DATA_OUTS["S"],
        preprocessing=PREPROCESSING,
    script:
        "scripts/prepare_data.py"


rule make_embedded_datasets:
    input:
        rules.make_text_datasets.output,
    params:
        outdir=DATA_OUTS["E"],
    output:
        [directory(f"{DATA_OUTS['E']}/{d}") for d in PREPROCESSING.keys()],
    script:
        "scripts/prepare_data.py"


rule pool_embeddings:
    input:
        rules.make_embedded_datasets.output,
    params:
        outdir=DATA_OUTS["P"],
    output:
        *[directory(d) for d in rules.all.input.pooled],
    script:
        "scripts/prepare_data.py"
