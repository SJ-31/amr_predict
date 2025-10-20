include: "Snakefile"


from pathlib import Path

OUTDIRS = {"cv": f"{OUT}/evaluation/cv", "holdout": f"{OUT}/evaluation/holdout"}
DEVICE = "cuda"
DATASETS = list(Path(f"{REMOTE}/datasets/pooled/{IN_DATE}").iterdir())

if TEST:
    DEVICE = "cpu"
    config["tasks"]["regression"] = ["AMK", "GEN"]
    config["cross_validate"]["k_fold"]["n_splits"] = 2
    config["cross_validate"]["models"] = ["baseline"]

cv_results = expand(
    "{o}/{m}/{d}_{t}.csv",
    o=OUTDIRS["cv"],
    m=config["cross_validate"]["models"],
    d=[d.stem for d in DATASETS],
    t=["regression", "classification"],
)

RESULTS = {
    "cv": {
        "cv_r": list(filter(lambda x: x.endswith("_regression.csv"), cv_results)),
        "cv_c": list(filter(lambda x: x.endswith("_classification.csv"), cv_results)),
    }
    # "holdout"
}

if TEST:
    del RESULTS["cv"]["cv_c"]


rule all:
    input:
        **RESULTS["cv"],


rule cross_validate:
    params:
        datasets=DATASETS,
        device=DEVICE,
        outdir=OUTDIRS["cv"],
    output:
        **RESULTS["cv"],
    script:
        "scripts/evaluate.py"
