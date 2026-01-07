include: "Snakefile"


configfile: "models.yaml"


from amr_predict.preprocessing import EMBEDDING_METHODS
from typing import get_args


from pathlib import Path

if TEST:
    IN_DATE = "for_interpret"
    config["train_sae"]["token-level"]["run"] = False
    config["train_sae"]["sequence-level"]["n"] = 9
    config["train_sae"]["sequence-level"]["n_sequence"] = 3
    config["train_sae"]["genome-level"]["n"] = 9
    config["train_sae"]["expansion_factor"] = 3
    config["train_sae"]["trainer"]["max_epochs"] = 3
    config["embedding"] = "seqLens"
    config["train_sae"]["dataloader"]["batch_size"] = 3


def default_log(rule_name):
    return f"{LOGDIR}/interpret-{rule_name}.log"


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

MODELS = {}  # Mapping of model file name to the dataset containing the sample metadata
for group, paths in DATASETS.items():
    for path in paths:
        if group == "pooled" and "genome-level" in LEVELS:
            MODELS[f"genome-level_{path.stem}.pth"] = path
        if group == "text" and "token-level" in LEVELS:
            MODELS[f"token-level_{path.stem}.pth"] = path
        if group == "text" and "sequence-level" in LEVELS:
            MODELS[f"sequence-level_{path.stem}.pth"] = path


def sae_plotting_paths(intermediate, as_dir):
    dataset_names = [Path(v).stem for v in MODELS.values()]
    ex = expand("{o}/{i}/{s}", o=OUTDIRS["sae"], i=intermediate, s=dataset_names)
    if as_dir:
        return directory(ex)
    return ex


# * Rules


rule all:
    input:
        models=expand("{o}/models/{m}", o=OUTDIRS["sae"], m=MODELS.keys()),


rule train_sae:
    input:
        **DATASETS,
    output:
        models=rules.all.input.models,
        # TODO: maybe add in the metrics as well?
    params:
        outdir=OUTDIRS["sae"],
        caches=Path(f"{OUT}/{IN_DATE}/datasets/embedded"),
        pooled=Path(f"{OUT}/{IN_DATE}/datasets/pooled"),
    resources:
        **GPU20,
    log:
        default_log("train_sae"),
    script:
        "scripts/interpret.py"


rule eval_sae:
    input:
        rules.train_sae.output.models,
    output:
        latent_summary_data=f"{OUTDIRS['sae']}/latent_summary.csv",
        concept_scoring_data=f"{OUTDIRS['sae']}/concept_scoring.csv",
        latent_summary_plot=f"{OUTDIRS['sae']}/latent_summary.png",
        activation_plots=sae_plotting_paths("activation_plots", True),
        umaps=sae_plotting_paths("latent_umap", True),
    params:
        caches=f"{OUT}/{IN_DATE}/datasets/embedded",
        model_dict=MODELS,
        outdir=OUTDIRS["sae"],
        pooled=f"{OUT}/{IN_DATE}/datasets/pooled",
    log:
        default_log("eval_sae"),
    script:
        "scripts/interpret.py"
