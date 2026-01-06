include: "Snakefile"


configfile: "models.yaml"


from amr_predict.preprocessing import EMBEDDING_METHODS
from typing import get_args


from pathlib import Path

if TEST:
    IN_DATE = "esm_test"
    config["train_sae"]["token-level"]["run"] = False


OUTDIRS = {
    "sae": f"{OUT}/{DATE}/sae",
}
DEVICE = config.get("device", "cuda")

dpath = Path(f"{REMOTE}/{IN_DATE}/datasets")


DATASETS = {"text": list(dpath.joinpath("processed_sequences").iterdir()), "pooled": []}
for d in dpath.joinpath("pooled").iterdir():
    name = d.stem.split("-", 1)[0]
    if (config["preprocessing"][name]).get("method") not in get_args(EMBEDDING_METHODS):
        DATASETS["pooled"].append(d)


LEVELS = [
    s
    for s in ("sequence-level", "token-level", "genome-level")
    if config["train_sae"][s]["run"]
]

MODELS = {}
for group, paths in DATASETS.items():
    for path in paths:
        if group == "pooled":
            MODELS[f"genome-level_{path}.pth"] = path
        else:
            if group == "text" and "token-level" in LEVELS:
                MODELS[f"token-level_{path}.pth"] = path
            if group == "text" and "sequence-level" in LEVELS:
                MODELS[f"sequence-level_{path}.pth"] = path


rule all:
    input:
        models=expand("{o}/models/{m}", o=OUTDIRS["sae"], m=MODELS.keys()),


rule train_sae:
    input:
        **DATASETS,
    output:
        models=all.input.models,
        # TODO: maybe add in the metrics as well?
    params:
        outdir=OUTDIRS["sae"],
        caches=f"{OUT}/{IN_DATE}/datasets/embedded",
        pooled=f"{OUT}/{IN_DATE}/datasets/pooled",


rule eval_sae:
    input:
        train_sae.output.models,
    output:
        latent_summary_data=f"{OUTDIRS['sae']}/latent_summary.csv",
        concept_scoring_data=f"{OUTDIRS['sae']}/concept_scoring.csv",
        latent_summary_plot=f"{OUTDIRS['sae']}/latent_summary.png",
        activation_plots=directory(
            expand("{o}/activation_plots/{m}", o=OUTDIRS["sae"], m=all.input.models)
        ),
    params:
        model_dict=MODELS,
        outdir=OUTDIRS["sae"],
        pooled=f"{OUT}/{IN_DATE}/datasets/pooled",
        sequences=f"{OUT}/{IN_DATE}/datasets/processed_sequences",
