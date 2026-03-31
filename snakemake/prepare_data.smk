from pathlib import Path
from typing import get_args
from amr_predict.preprocessing import EMBEDDING_METHODS


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
    config["pool_embeddings"]["methods"] = {
        "mean": {},
        "sum": {},
        "similarity": {},
    }
    config["sample_metadata"][
        "file"
    ] = f"{config["data"]["meta"]}/combined_sample_meta.tsv"
    config["sample_metadata"]["id_col"] = "sample"
    del config["preprocessing"]["baseline-feature_presence_bakta"]

PREPROCESSING = config["preprocessing"]
EMBEDDING = config["embedding"]


OUTDIRS = {
    k: f"{REMOTE}/{DATE}/datasets/{s}"
    for k, s in zip(
        ["S", "E", "P"],
        ["processed_sequences", "embedded", "pooled"],
        # Datasets are...
        # 1. Processed genome data e.g. split into ORFs, binned etc.
        # 2. Sequences embedded by the chosen GLM
        # 3. Sequences pooled into genome-level representations
    )
}
pooling_methods = [k for k in config["pool_embeddings"]["methods"].keys()]
EMBEDDING_RES = GPU40.copy() if not TEST else GPU20.copy()
if config["embedding"] == "esm":
    tmp = {}
    # Only use esm for pure ORFs
    for k, v in PREPROCESSING.items():
        if (v.get("method") in get_args(EMBEDDING_METHODS)) or (
            v.get("split_method") == "bakta"
            and v.get("utr_amount") is None
            and v.get("upstream_context") is None
            and v.get("downstream_context") is None
        ):
            tmp[k] = v
    PREPROCESSING = tmp
elif config["embedding"] == "Evo2":
    EMBEDDING_RES = {"qos": "cpu24h", "mem": "30G"}
EMBEDDING_RES["time"] = config.get("slurm_time_limit", "15-0:0:0")

PLOT_OUT = Path(f"{OUT}/{DATE}/embedding_comparison/pooled_distance_correlation")

POOLED_ALREADY = [
    d
    for d, k in PREPROCESSING.items()
    if k.get("method") in get_args(EMBEDDING_METHODS)
]
TO_POOL = [d for d in PREPROCESSING.keys() if d not in POOLED_ALREADY]
POOLED_PLOTS = expand(
    "{o}/{d}-{e}-{p}.png", o=PLOT_OUT, d=TO_POOL, e=EMBEDDING, p=pooling_methods
)


CACHE_CHECKS = [f"{OUTDIRS['E']}/{d}-{EMBEDDING}_cache.complete" for d in TO_POOL]

print(f"INFO: Retrieving genomes from `{config["genomes"]}`")


rule all:
    input:
        expand(
            "{o}/{d}-{e}-{p}.png",
            o=PLOT_OUT,
            d=TO_POOL,
            e=EMBEDDING,
            p=pooling_methods,
        ),
        embedded=CACHE_CHECKS,
        pooled=expand(
            f"{OUTDIRS['P']}/{{d}}-{EMBEDDING}-{{p}}", d=TO_POOL, p=pooling_methods
        ),
        other_pooled=[f"{OUTDIRS['P']}/{d}" for d in POOLED_ALREADY],
        meta=f"{REMOTE}/{DATE}/seq_metadata.csv",


rule get_seq_metadata:
    output:
        f"{REMOTE}/{DATE}/seq_metadata.csv",
    log:
        log=f"{LOGDIR}/prepare_data/get_seq_metadata.log",
        profile=f"{LOGDIR}/prepare_data/get_seq_metadata_mem.bin",
    script:
        "scripts/prepare_data.py"


rule make_text_dataset:
    output:
        directory(f"{OUTDIRS['S']}/{{dataset}}"),
    input:
        rules.get_seq_metadata.output,
    log:
        log=f"{LOGDIR}/prepare_data/make_text_dataset-{{dataset}}.log",
        profile=f"{LOGDIR}/prepare_data/make_text_dataset-{{dataset}}_mem.bin",
    resources:
        **BIG_MEM,
    params:
        preprocessing=PREPROCESSING,
    script:
        "scripts/prepare_data.py"


rule make_baseline:
    output:
        directory(f"{OUTDIRS['P']}/{{dataset}}"),
    wildcard_constraints:
        dataset="baseline-.*",
    input:
        rules.get_seq_metadata.output,
    resources:
        **BIG_MEM,
    log:
        log=f"{LOGDIR}/prepare_data/make_baseline-{{dataset}}.log",
        profile=f"{LOGDIR}/prepare_data/make_baseline-{{dataset}}_mem.bin",
    params:
        preprocessing=PREPROCESSING,
    script:
        "scripts/prepare_data.py"


rule make_embedded_dataset:
    input:
        f"{OUTDIRS['S']}/{{dataset}}",
    log:
        log=f"{LOGDIR}/prepare_data/make_embedded_dataset-{{dataset}}.log",
        profile=f"{LOGDIR}/prepare_data/make_embedded_dataset-{{dataset}}_mem.bin",
    resources:
        **EMBEDDING_RES,
    params:
        outdir=OUTDIRS["E"],
    output:
        f"{OUTDIRS['E']}/{{dataset}}-{EMBEDDING}_cache.complete",
    script:
        "scripts/prepare_data.py"


rule pool_embeddings:
    input:
        lambda wc: f"{OUTDIRS['E']}/{wc.get('dataset')}-{EMBEDDING}_cache.complete",
    params:
        outdir=OUTDIRS["P"],
        textdir=OUTDIRS["S"],
        plotdir=PLOT_OUT,
        pooling=lambda wc: wc.get("pooling"),
    resources:
        **BIG_MEM,
    log:
        log=f"{LOGDIR}/prepare_data/pool_embeddings-{{dataset}}-{{pooling}}.log",
        profile=f"{LOGDIR}/prepare_data/pool_embeddings-{{dataset}}-{{pooling}}_mem.bin",
    output:
        dataset=directory(f"{OUTDIRS['P']}/{{dataset}}-{EMBEDDING}-{{pooling}}"),
        plot=expand(
            "{p}/{{dataset}}-{e}-{{pooling}}.{ext}",
            p=PLOT_OUT,
            e=EMBEDDING,
            ext=["png", "csv"],
        ),
    script:
        "scripts/prepare_data.py"
