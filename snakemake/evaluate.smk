include: "Snakefile"


from pathlib import Path

OUTDIRS = {"cv": f"{OUT}/evaluation/cv", "holdout": f"{OUT}/evaluation/holdout"}

DATASETS = list(Path(f"{REMOTE}/datasets/pooled/{IN_DATE}").iterdir())
RESULTS = {
    "cv": {
        "cv_r": [f"{OUTDIRS['cv']}/{d.stem}_regression.csv" for d in DATASETS],
        "cv_c": [f"{OUTDIRS['cv']}/{d.stem}_classification.csv" for d in DATASETS],
    }
    # "holdout"
}
DEVICE = "cuda"

if TEST:
    DEVICE = "cpu"
    config["tasks"]["regression"] = ["AMK", "GEN"]
    del RESULTS["cv"]["cv_c"]
    config["cross_validate"]["k_fold"]["n_splits"] = 2


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
