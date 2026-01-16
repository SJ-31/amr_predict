include: "Snakefile"


# Passed test [2026-01-07 Wed]
if TEST:
    config["compare_embeddings"]["cluster_on"] = ["sample", "hamr_gene_symbol"]
    config["compare_pooled"]["cluster_on"] = ["amikacin_class", "genus"]
    config["compare_pooled"]["continuous"] = ["amikacin", "gentamicin", "imipenem"]
    config["compare_pooled"]["pair_distance_distribution"]["cols"] = [
        "amikacin_class",
        "genus",
    ]
    config["compare_pooled"]["covariate_distance_correlation"]["cols"] = [
        "amikacin",
        "imipenem",
    ]
    config["compare_pooled"]["neighbor_proportion"]["cols"] = ["genus", "family"]
    config["compare_pooled"]["pair_distance_distribution"]["kws"]["replace"] = True
    config["compare_embeddings"]["bootstrap_rounds"] = 2

DIRS2RULE = {"sequences": "compare_embeddings", "pooled": "compare_pooled"}
OUTDIRS = {k: f"{OUT}/{DATE}/embedding_comparison/{k}" for k in DIRS2RULE.keys()}
DATASETS = {
    "sequences": Path(f"{REMOTE}/{IN_DATE}/datasets/processed_sequences"),
    "pooled": Path(f"{REMOTE}/{IN_DATE}/datasets/pooled"),
}


RESULTS = {}
for k, r in DIRS2RULE.items():
    dnames = [d.stem for d in DATASETS[k].iterdir()]
    RESULTS[f"{k}_plots_c"] = expand("{o}/plots-c/{d}", o=OUTDIRS[k], d=dnames)
    RESULTS[f"{k}_plots_d"] = expand("{o}/plots-d/{d}", o=OUTDIRS[k], d=dnames)
    RESULTS[f"{k}_metrics"] = expand(f"{OUTDIRS[k]}/{{d}}_metrics.csv", d=dnames)


rule all:
    input:
        **RESULTS,


for key, rname in DIRS2RULE.items():

    rule:
        name:
            rname
        input:
            f"{DATASETS[key]}/{{dataset}}",
        output:
            metrics=f"{OUTDIRS[key]}/{{dataset}}_metrics.csv",
            plots_c=directory(f"{OUTDIRS[key]}/plots-c/{{dataset}}"),
            plots_d=directory(f"{OUTDIRS[key]}/plots-d/{{dataset}}"),
        log:
            f"{LOGDIR}/compare_embeddings/{rname}_{{dataset}}.log",
        params:
            caches=Path(f"{REMOTE}/{IN_DATE}/datasets/embedded"),
            outdir=OUTDIRS[key],
        script:
            "scripts/compare_embeddings.py"
