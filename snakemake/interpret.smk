include: "Snakefile"


configfile: "models.yaml"


from pathlib import Path

OUTDIRS = {
    "sae": f"{OUT}/{DATE}/sae",
    "holdout": f"{OUT}/{DATE}/evaluation/holdout",
}
DEVICE = config.get("device", "cuda")

dpath = Path(f"{REMOTE}/{IN_DATE}/datasets")

DATASETS = {
    "text": list(dpath.joinpath("processed_sequences").iterdir()),
    "pooled": list(dpath.joinpath("pooled").iterdir()),
}
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
        caches="",
        pooled="",


rule eval_sae:
    input:
        train_sae.output.models,
    params:
        model_dict=MODELS,
