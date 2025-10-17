import itertools


include: "Snakefile"


if TEST:
    config["compare_embeddings"]["cluster_on"] = [
        "sample",
        "Strain",
    ]
    config["compare_pooled"]["cluster_on"] = ["AMK_class", "GEN_class"]
    config["compare_pooled"]["continuous"] = ["AMK", "CRO", "IPM"]

to_compare = ["sequences", "pooled"]

OUTDIRS = {
    k: f"{OUT}/embedding_comparison/{DATE}/{v}" for k, v in zip(["S", "P"], to_compare)
}

DATASETS = {
    "S": list(Path(f"{REMOTE}/datasets/embedded/{IN_DATE}").iterdir()),
    "P": list(Path(f"{REMOTE}/datasets/pooled/{IN_DATE}").iterdir()),
}


RESULTS = {}
for k, v, r in zip(["S", "P"], to_compare, ["compare_embeddings", "compare_pooled"]):
    RESULTS[k] = {
        "plots": expand(
            "{o}/plots/{i}_{d}_{p}{s}.png",
            o=OUTDIRS[k],
            d=[d.stem for d in DATASETS[k]],
            p=["pca", "umap"],
            s=["-d", "-c"] if k == "P" else [""],
            i=range(config[r]["bootstrap_rounds"]),
        ),
        "metrics": f"{OUTDIRS[k]}/metrics.csv",
    }


rule all:
    input:
        list(
            itertools.chain.from_iterable(
                [
                    list(RESULTS[x]["plots"]) + [RESULTS[x]["metrics"]]
                    for x in ["S", "P"]
                ]
            )
        ),


rule compare_embeddings:
    params:
        datasets=DATASETS["S"],
        outdir=OUTDIRS["S"],
    output:
        metrics=RESULTS["S"]["metrics"],
        plots=RESULTS["S"]["plots"],
    script:
        "scripts/reporting.py"


rule compare_pooled:
    params:
        datasets=DATASETS["P"],
        outdir=OUTDIRS["P"],
    output:
        metrics=RESULTS["P"]["metrics"],
        plots=RESULTS["P"]["plots"],
    script:
        "scripts/reporting.py"
