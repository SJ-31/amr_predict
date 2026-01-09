include: "Snakefile"


configfile: "models.yaml"


from pathlib import Path

IN = Path(f"{REMOTE}/{IN_DATE}/datasets")

OUTDIRS = {
    "cv": f"{OUT}/{DATE}/evaluation/cv",
    "holdout": f"{OUT}/{DATE}/evaluation/holdout",
    "cv_ctrl": f"{OUT}/{DATE}/evaluation/cv_control",
}
DEVICE = config.get("device", "cuda")
DATASETS = list(IN / "pooled").iterdir()
for sae_data in ("reconstructed", "sae_activations"):
    sd_path = IN / sae_data
    if sd_path.exists():
        DATASETS.extend(list(sae_recon.iterdir()))


def default_log(rule_name):
    return f"{LOGDIR}/evaluate-{rule_name}.log"


if TEST:
    DEVICE = "cpu"
    config["tasks"]["regression"] = ["AMK", "GEN"]
    # config["tasks"]["classification"] = ["IPM_class", "GEN_class"]
    config["cross_validate"]["k_fold"]["n_splits"] = 2
    config["cross_validate"]["models"] = ["baseline", "mlp"]
    config["dataloader"]["batch_size"] = 20
    config["holdout"]["models"] = ["baseline", "mlp"]
    config["holdout"]["splits"] = None
    # config["holdout"]["splits"] = {
    #     "test1": {
    #         "sample": {
    #             "SAMN29490345": "EXACT",
    #             "SAMN29490346": "EXACT",
    #             "SAMN29490347": "EXACT",
    #         }
    #     },
    #     "test2": {
    #         "sample": {
    #             "SAMN29490348": "EXACT",
    #             "SAMN29490350": "EXACT",
    #             "SAMN29490351": "EXACT",
    #         }
    #     },
    # }

all_results = expand(
    "{o}/{m}/{d}_{t}.csv",
    o=[OUTDIRS["cv"], OUTDIRS["holdout"], OUTDIRS["cv_ctrl"]],
    m=config["cross_validate"]["models"],
    d=[d.stem for d in DATASETS],
    t=["regression", "classification"],
)

cv_results = list(filter(lambda x: x.startswith(OUTDIRS["cv"]), all_results))
cv_ctrl_results = list(filter(lambda x: x.startswith(OUTDIRS["cv_ctrl"]), all_results))
holdout_results = list(filter(lambda x: x.startswith(OUTDIRS["holdout"]), all_results))

RESULTS = {
    "cv": {
        "cv_r": list(filter(lambda x: x.endswith("_regression.csv"), cv_results)),
        "cv_c": list(filter(lambda x: x.endswith("_classification.csv"), cv_results)),
    },
    "cv_ctrl": {
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
            **RESULTS["cv_ctrl"],

else:

    rule all:
        input:
            **RESULTS["cv"],
            **RESULTS["cv_ctrl"],
            **RESULTS["holdout"],


rule cross_validate:
    params:
        datasets=DATASETS,
        device=DEVICE,
        outdir=OUTDIRS["cv"],
    log:
        default_log("cross_validate"),
    output:
        **RESULTS["cv"],
    script:
        "scripts/evaluate.py"


rule cv_control_tasks:
    params:
        datasets=DATASETS,
        device=DEVICE,
        outdir=OUTDIRS["cv_ctrl"],
    log:
        default_log("cross_validate_control"),
    output:
        **RESULTS["cv_ctrl"],
    script:
        "scripts/evaluate.py"


rule holdout:
    params:
        datasets=DATASETS,
        device=DEVICE,
        outdir=OUTDIRS["holdout"],
    log:
        default_log("holdout"),
    output:
        **RESULTS["holdout"],
    script:
        "scripts/evaluate.py"
