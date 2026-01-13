include: "Snakefile"


configfile: "models.yaml"


from pathlib import Path

IN = Path(f"{REMOTE}/{IN_DATE}/datasets")

OUTDIRS = {
    "cv": f"{OUT}/{DATE}/evaluation/cv",
    "holdout": f"{OUT}/{DATE}/evaluation/holdout",
    "ctrl_cv": f"{OUT}/{DATE}/evaluation/ctrl_cv",
}
DEVICE = config.get("device", "cuda")
DATASETS = list((IN / "pooled").iterdir())
for sae_data in ("reconstructed", "sae_activations"):
    sd_path = IN / sae_data
    if sd_path.exists():
        DATASETS.extend(list(sae_recon.iterdir()))

GROUP_MAPPING = {
    "ctrl_cv": "Cross-validation (control tasks)",
    "cv": "Cross-validation",
}
METRICS = {
    "classification": ("kappa", "acc", "mcc", "auroc", "aupr"),
    "regression": ("mse", "spearman", "pearson", "nrmse"),
}


def default_log(rule_name):
    return f"{LOGDIR}/evaluate-{rule_name}.log"


def report_figures(outdir):
    result = {}
    file_mapping = {}
    for group in ("cv", "ctrl_cv", "holdout"):
        if group == "holdout" and not config["holdout"]["splits"]:
            continue
        for task in ("regression", "classification"):
            if not config["tasks"][task] or task == "regression" and group == "ctrl_cv":
                continue
            key = f"{group}_{task}"
            result[key] = report(
                directory(f"{outdir}/.{group}_{task}"),
                patterns=["*.png"],
                category=GROUP_MAPPING.get(group, group),
                subcategory=task,
            )

    return result


if TEST:
    # DEVICE = "cpu"
    config["tasks"]["regression"] = ["amikacin", "gentamicin"]
    config["tasks"]["classification"] = ["amikacin_class", "gentamicin_class"]
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
    o=[OUTDIRS["cv"], OUTDIRS["holdout"], OUTDIRS["ctrl_cv"]],
    m=config["cross_validate"]["models"],
    d=[d.stem for d in DATASETS],
    t=["regression", "classification"],
)

RESULTS = {}
for k, v in OUTDIRS.items():
    for task in ("regression", "classification"):
        RESULTS[f"{k}_{task[0]}"] = list(
            filter(
                lambda x: x.startswith(v) and x.endswith(f"_{task}.csv"), all_results
            )
        )
del RESULTS["ctrl_cv_r"]
if not config["holdout"]["splits"]:
    RESULTS = {k: v for k, v in RESULTS.items() if not k.startswith("holdout")}


rule summarize_results:
    input:
        **RESULTS,
    log:
        default_log("report"),
    params:
        outdir=Path(f"{OUT}/{DATE}/evaluation"),
    output:
        **report_figures(f"{OUT}/{DATE}/evaluation"),
    script:
        "scripts/evaluate.py"


for model in config["cross_validate"]["models"]:
    for dataset in DATASETS:
        name = dataset.stem
        default_params = {"model": model}
        if model == "baseline":
            res = BIG_MEM
            default_params["device"] = "cpu"
        else:
            res = GPU20
            default_params["device"] = DEVICE

        def get_output(task):
            return {
                "regression": f"{OUTDIRS[task]}/{model}/{name}_regression.csv",
                "classification": f"{OUTDIRS[task]}/{model}/{name}_classification.csv",
            }

        rule:
            name:
                f"cv-{model}_{name}"
            params:
                **default_params,
            log:
                default_log(f"cv-{model}_{name}"),
            input:
                dataset,
            resources:
                **res,
            output:
                **get_output("cv"),
            script:
                "scripts/evaluate.py"

        rule:
            name:
                f"cv_control_task-{model}_{name}"
            params:
                **default_params,
            input:
                dataset,
            resources:
                **res,
            output:
                classification=get_output("ctrl_cv")["classification"],
            log:
                default_log(f"ctrl_cv-{model}_{name}"),
            script:
                "scripts/evaluate.py"

        if config["holdout"]["splits"]:

            rule:
                name:
                    f"holdout-{model}_{name}"
                params:
                    **default_params,
                input:
                    dataset,
                resources:
                    **res,
                log:
                    default_log(f"holdout-{model}_{name}"),
                output:
                    **get_output("ctrl_cv"),
                script:
                    "scripts/evaluate.py"


# TODO: [2026-01-08 Thu] need to test that this works
# TODO: would like a rule that aggregates all the results across everything
