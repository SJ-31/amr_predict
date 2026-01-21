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


def default_log(rule_name):
    return f"{LOGDIR}/evaluate/{rule_name}.log"


# [2026-01-13 Tue] passed
if TEST:
    config["tasks"]["regression"] = ["amikacin", "gentamicin"]
    config["tasks"]["classification"] = ["amikacin_class", "gentamicin_class"]
    config["cross_validate"]["k_fold"]["n_splits"] = 2
    config["cross_validate"]["models"] = ["baseline", "mlp"]
    config["dataloader"]["batch_size"] = 20
    config["holdout"]["models"] = ["baseline", "mlp"]
    config["holdout"]["validation_kws"] = {}
    config["holdout"]["splits"] = {
        "test1": {"species": {e: "EXACT" for e in "DE"}},
        "test2": {"genus": {e: "EXACT" for e in "AB"}},
    }
    config["holdout"]["bootstrap_splits"] = {
        "test_bootstrap": {
            "repeat": 3,
            "kws": {"train_size": 0.5, "shuffle": True},
            "exclude": {"species": {e: "EXACT" for e in "DE"}},
        }
    }


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


rule all:
    input:
        **RESULTS,


# TODO: unfinished - could you re-write the below using wildcards?
# rule cv:
#     output:
#         "{outdir}/{model}/{dataset}_{task}.csv",
#     params:
#         datasets={d.stem: d for d in DATASETS},
#         model=lambda wc: wc.get("model"),
#         device=lambda wc: "cpu" if wc.get("model") == "baseline" else DEVICE,
#     shell:
#         "echo {outdir} {model}"


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
                    **get_output("holdout"),
                script:
                    "scripts/evaluate.py"


# TODO: [2026-01-08 Thu] need to test that this works
# TODO: would like a rule that aggregates all the results across everything
