include: "Snakefile"


from pathlib import Path

OUTDIRS = {
    "cv": f"{OUT}/{DATE}/evaluation/cv",
    "holdout": f"{OUT}/{DATE}/evaluation/holdout",
}
DEVICE = config.get("device", "cuda")
DATASETS = list(Path(f"{REMOTE}/{IN_DATE}/datasets/pooled").iterdir())

if TEST:
    DEVICE = "cpu"
    config["tasks"]["regression"] = ["AMK", "GEN"]
    config["cross_validate"]["k_fold"]["n_splits"] = 2
    config["cross_validate"]["models"] = ["baseline"]
    config["holdout"]["models"] = ["baseline"]
    config["holdout"]["splits"] = {
        "test1": {
            "sample": {
                "SAMN29490345": "EXACT",
                "SAMN29490346": "EXACT",
                "SAMN29490347": "EXACT",
            }
        },
        "test2": {
            "sample": {
                "SAMN29490348": "EXACT",
                "SAMN29490350": "EXACT",
                "SAMN29490351": "EXACT",
            }
        },
    }

all_results = expand(
    "{o}/{m}/{d}_{t}.csv",
    o=[OUTDIRS["cv"], OUTDIRS["holdout"]],
    m=config["cross_validate"]["models"],
    d=[d.stem for d in DATASETS],
    t=["regression", "classification"],
)

cv_results = list(filter(lambda x: x.startswith(OUTDIRS["cv"]), all_results))
holdout_results = list(filter(lambda x: x.startswith(OUTDIRS["holdout"]), all_results))

RESULTS = {
    "cv": {
        "cv_r": list(filter(lambda x: x.endswith("_regression.csv"), cv_results)),
        "cv_c": list(filter(lambda x: x.endswith("_classification.csv"), cv_results)),
    },
    "holdout": {
        "holdout_r": list(
            filter(lambda x: x.endswith("_regression.csv"), holdout_results)
        ),
        "holdout_c": list(
            filter(lambda x: x.endswith("_classification.csv"), holdout_results)
        ),
    },
}

if TEST:
    del RESULTS["cv"]["cv_c"]
    del RESULTS["holdout"]["holdout_c"]


if not config["holdout"]["splits"]:

    rule all:
        input:
            **RESULTS["cv"],

else:

    rule all:
        input:
            **RESULTS["cv"],
            **RESULTS["holdout"],


rule cross_validate:
    params:
        datasets=DATASETS,
        device=DEVICE,
        outdir=OUTDIRS["cv"],
    output:
        **RESULTS["cv"],
    script:
        "scripts/evaluate.py"


rule holdout:
    params:
        datasets=DATASETS,
        device=DEVICE,
        outdir=OUTDIRS["holdout"],
    output:
        **RESULTS["holdout"],
    script:
        "scripts/evaluate.py"
