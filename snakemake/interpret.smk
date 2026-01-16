include: "Snakefile"


configfile: "models.yaml"


from amr_predict.preprocessing import EMBEDDING_METHODS
from typing import get_args


from pathlib import Path

if TEST:
    config["train_sae"]["token-level"]["run"] = False
    config["train_sae"]["sequence-level"]["n"] = 9
    config["train_sae"]["sequence-level"]["n_sequence"] = 3
    config["train_sae"]["genome-level"]["n"] = 9
    config["train_sae"]["expansion_factor"] = 3
    config["train_sae"]["trainer"]["max_epochs"] = 3
    config["train_sae"]["dataloader"]["batch_size"] = 3
    config["save_activations"]["genome-level"]["n"] = 5
    config["reconstruct_datasets"]["genome-level"]["n"] = 5
    config["save_activations"]["sequence-level"]["n"] = 9
    config["save_activations"]["sequence-level"]["n_sequence"] = 3
    config["reconstruct_datasets"]["sequence-level"]["n"] = 9
    config["reconstruct_datasets"]["sequence-level"]["n_sequence"] = 3
    config["eval_sae"]["concept_cols"]["sequence-level"].remove("any_resistant")
    config["eval_sae"]["concept_cols"]["sequence-level"].remove("any_susceptible")
    config["eval_sae"]["concept_cols"]["sequence-level"].remove("in_gene")
    config["eval_sae"]["concept_cols"]["genome-level"].remove("any_resistant")
    config["eval_sae"]["concept_cols"]["genome-level"].remove("any_susceptible")


def default_log(rule_name):
    return f"{LOGDIR}/interpret-{rule_name}.log"


OUTDIRS = {"sae": f"{REMOTE}/{DATE}/sae", "datasets": f"{REMOTE}/{DATE}/datasets"}
OUTDIRS["save_activations"] = Path(OUTDIRS["datasets"]) / "sae_activations"
OUTDIRS["reconstruct_datasets"] = Path(OUTDIRS["datasets"]) / "reconstructed"

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

# TODO: rename MODELS to WEIGHTS in the variable and as output

WEIGHTS = {}
# Mapping of weight file name to the dataset containing the sample metadata
RECONSTRUCTIONS = {"path": Path(OUTDIRS["datasets"]) / "reconstructed", "list": []}
ACTIVATIONS = {"path": Path(OUTDIRS["datasets"]) / "sae_activations", "list": []}


def add_sae_saved(level, path):
    if level != "genome-level":
        RECONSTRUCTIONS["list"].append(
            RECONSTRUCTIONS["path"] / f"recon_{level}_{path.stem}"
        )
        ACTIVATIONS["list"].append(ACTIVATIONS["path"] / f"sae-act_{level}_{path.stem}")
    else:
        RECONSTRUCTIONS["list"].append(RECONSTRUCTIONS["path"] / f"recon_{path.stem}")
        ACTIVATIONS["list"].append(ACTIVATIONS["path"] / f"sae-act_{path.stem}")


# For each dataset, train SAE at different embedding levels
for group, paths in DATASETS.items():
    for path in paths:
        if group == "pooled" and "genome-level" in LEVELS:
            WEIGHTS[f"genome-level_{path.stem}.pth"] = path
            add_sae_saved("genome-level", path)
        if group == "text" and "token-level" in LEVELS:
            WEIGHTS[f"token-level_{path.stem}.pth"] = path
            add_sae_saved("token-level", path)
        if group == "text" and "sequence-level" in LEVELS:
            WEIGHTS[f"sequence-level_{path.stem}.pth"] = path
            add_sae_saved("sequence-level", path)


def sae_plotting_paths(intermediate, as_dir):
    dataset_names = [Path(v).stem for v in WEIGHTS.values()]
    ex = expand("{o}/{i}/{s}", o=OUTDIRS["sae"], i=intermediate, s=dataset_names)
    if as_dir:
        return directory(ex)
    return ex


SAE_EVALS = {
    "latent_summary_data": f"{OUTDIRS['sae']}/latent_summary.csv",
    "concept_scoring_data": f"{OUTDIRS['sae']}/concept_scoring.csv",
    "latent_summary_plot": f"{OUTDIRS['sae']}/latent_summary.png",
}

# * Rules


rule all:
    input:
        **SAE_EVALS,
        reconstructions=RECONSTRUCTIONS["list"],
        activations=ACTIVATIONS["list"],
        weights=expand("{o}/weights/{m}", o=OUTDIRS["sae"], m=WEIGHTS.keys()),
        activation_plots=sae_plotting_paths("activation_plots", False),
        umaps=sae_plotting_paths("latent_umap", False),


rule train_sae:
    input:
        lambda wc: WEIGHTS[f'{wc.get("level")}_{wc.get("dataset")}.pth'],
    output:
        f"{OUTDIRS['sae']}/weights/{{level,[a-z]+-level}}_{{dataset}}.pth",
    params:
        level=lambda wc: wc.get("level"),
        outdir=Path(OUTDIRS["sae"]),
        caches=Path(f"{dpath}/embedded"),
        pooled=Path(f"{dpath}/pooled"),
    resources:
        **GPU20,
    log:
        f"{LOGDIR}/interpret-train_sae_{{level}}_{{dataset}}.log",
    script:
        "scripts/interpret.py"


# rule save_activations:
#     input:
#         rules.all.input.weights,
#     params:
#         caches=Path(f"{dpath}/embedded"),
#         outdir=ACTIVATIONS["path"],
#         weight_dict=WEIGHTS,
#         pooled=Path(f"{dpath}/pooled"),
#     output:
#         *directory(ACTIVATIONS["list"]),
#     log:
#         default_log("save_activations"),
#     resources:
#         **GPU20,
#     script:
#         "scripts/interpret.py"


rule save_activations:
    input:
        rules.all.input.weights,
    params:
        caches=Path(f"{dpath}/embedded"),
        outdir=ACTIVATIONS["path"],
        weight_dict=WEIGHTS,
        pooled=Path(f"{dpath}/pooled"),
    output:
        *directory(ACTIVATIONS["list"]),
    log:
        default_log("save_activations"),
    resources:
        **GPU20,
    script:
        "scripts/interpret.py"


rule reconstruct_datasets:
    input:
        rules.all.input.models,
    params:
        caches=Path(f"{dpath}/embedded"),
        model_dict=MODELS,
        outdir=RECONSTRUCTIONS["path"],
        pooled=Path(f"{dpath}/pooled"),
    output:
        *directory(RECONSTRUCTIONS["list"]),
    resources:
        **GPU20,
    log:
        default_log("reconstruct"),
    script:
        "scripts/interpret.py"


rule eval_sae:
    input:
        rules.save_activations.output,
    output:
        **SAE_EVALS,
        activation_plots=sae_plotting_paths("activation_plots", True),
        umaps=sae_plotting_paths("latent_umap", True),
    params:
        caches=Path(f"{dpath}/embedded"),
        model_dict=MODELS,
        outdir=OUTDIRS["sae"],
        pooled=Path(f"{dpath}/pooled"),
        seqs=Path(f"{dpath}/processed_sequences"),
    log:
        default_log("eval_sae"),
    script:
        "scripts/interpret.py"
