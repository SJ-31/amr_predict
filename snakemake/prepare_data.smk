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
    config["pool_embeddings"]["methods"] = [
        {"method": m} for m in ("mean", "sum", "similarity")
    ]
    config["sample_metadata"][
        "file"
    ] = f"{config["data"]["meta"]}/combined_sample_meta.tsv"
    config["sample_metadata"]["id_col"] = "sample"
    to_keep = [[a, f"{a}_class"] for a in ["imipenem", "amikacin", "gentamicin"]]
    config["pool_embeddings"].update(
        {"obs_keep": [el for sublist in to_keep for el in sublist]}
    )
    del config["preprocessing"]["feature_presence_bakta"]

PREPROCESSING = config["preprocessing"]
EMBEDDING = config["embedding"]

DATA_OUTS = {
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
pooling_methods = [
    d.get("name", d["method"]) for d in config["pool_embeddings"]["methods"]
]
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

PLOT_OUT = f"{OUT}/{DATE}/embedding_comparison/pooled_distance_correlation"

POOLED_ALREADY = [
    d
    for d, k in PREPROCESSING.items()
    if k.get("method") in get_args(EMBEDDING_METHODS)
]
TO_POOL = [d for d in PREPROCESSING.keys() if d not in POOLED_ALREADY]
POOLED_PLOTS = expand(
    "{o}/{d}-{e}-{p}.png", o=PLOT_OUT, d=TO_POOL, e=EMBEDDING, p=pooling_methods
)

CACHE_CHECKS = ([f"{DATA_OUTS['E']}/{d}_{EMBEDDING}_cache.complete" for d in TO_POOL],)


def default_log(rule_name):
    return {
        "log": f"{LOGDIR}/prepare_data-{rule_name}.log",
        "profile": f"{LOGDIR}/prepare_data-{rule_name}_mem.bin",
    }


def get_pooled_out(as_dir: bool = False):
    expanded = expand(
        "{o}/{d}-{e}-{p}",
        o=DATA_OUTS["P"],
        e=EMBEDDING,
        d=TO_POOL,
        p=pooling_methods,
    )
    if as_dir:
        return directory(expanded)
    return expanded


rule all:
    input:
        POOLED_PLOTS,
        embedded=CACHE_CHECKS,
        pooled=get_pooled_out(),
        other_pooled=[f"{DATA_OUTS['P']}/{d}" for d in POOLED_ALREADY],
        meta=f"{PROCESSED}/{DATE}/seq_metadata.csv",


rule get_seq_metadata:
    output:
        rules.all.input.meta,
    log:
        **default_log("get_seq_metadata"),
    script:
        "scripts/prepare_data.py"


rule make_text_datasets:
    output:
        [directory(f"{DATA_OUTS['S']}/{d}") for d in TO_POOL],
        [directory(f"{DATA_OUTS['P']}/{d}") for d in POOLED_ALREADY],
    input:
        rules.get_seq_metadata.output,
    log:
        **default_log("make_text_datasets"),
    params:
        outdir=DATA_OUTS["S"],
        outdir_pooled=DATA_OUTS["P"],
        preprocessing=PREPROCESSING,
    script:
        "scripts/prepare_data.py"


rule make_embedded_datasets:
    input:
        rules.make_text_datasets.output,
    log:
        **default_log("make_embedded_datasets"),
    params:
        outdir=DATA_OUTS["E"],
        ignore=POOLED_ALREADY,
    output:
        CACHE_CHECKS,
    script:
        "scripts/prepare_data.py"


rule pool_embeddings:
    input:
        rules.make_embedded_datasets.output,
    params:
        outdir=DATA_OUTS["P"],
        textdir=DATA_OUTS["S"],
        plotdir=PLOT_OUT,
    log:
        **default_log("pool_embeddings"),
    output:
        get_pooled_out(True),
        POOLED_PLOTS,
    script:
        "scripts/prepare_data.py"
