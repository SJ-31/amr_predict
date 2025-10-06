include: "shared.smk"


OUTDIR = f"{OUT}/embedding_comparison/{DATE}"

EMBEDDINGS_IN = f"{REMOTE}/datasets/embedded/{IN_DATE}"

if TEST:
    config["compare_embeddings"]["cluster_on"] = [
        "sample",
        "resistance_mechanism",
        "Strain",
    ]


datasets = list(Path(EMBEDDINGS_IN).iterdir())

results = {
    "embedding_plots": expand(
        "{o}/plots/{i}_{d}_{p}.png",
        o=OUTDIR,
        d=[d.stem for d in datasets],
        p=["pca", "umap"],
        i=range(config["compare_embeddings"]["bootstrap_rounds"]),
    ),
    "embedding_metrics": f"{OUTDIR}/metrics.csv",
}


rule all:
    input:
        **results,


rule compare_embeddings:
    params:
        datasets=datasets,
        outdir=OUTDIR,
    output:
        metrics=results["embedding_metrics"],
        plots=results["embedding_plots"],
    script:
        "scripts/reporting.py"
