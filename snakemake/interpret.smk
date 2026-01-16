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
sae_gen = {
    "save_activations": ("sae-act", Path(OUTDIRS["datasets"]) / "sae_activations"),
    "reconstruct_datasets": ("recon", Path(OUTDIRS["datasets"]) / "reconstructed"),
}

DEVICE = config.get("device", "cuda")
DPATH = Path(f"{REMOTE}/{IN_DATE}/datasets")


DATASETS = {"text": list(DPATH.joinpath("processed_sequences").iterdir()), "pooled": []}
for d in DPATH.joinpath("pooled").iterdir():
    name = d.stem.split("-", 1)[0]
    if (config["preprocessing"][name]).get("method") not in get_args(EMBEDDING_METHODS):
        DATASETS["pooled"].append(d)

LEVELS = [
    s
    for s in ("sequence-level", "token-level", "genome-level")
    if config["train_sae"][s]["run"]
]

WEIGHTS = {}
# Mapping of weight file name to the dataset containing the sample metadata

# For each dataset, train SAE at different embedding levels
for group, paths in DATASETS.items():
    for path in paths:
        if group == "pooled" and "genome-level" in LEVELS:
            WEIGHTS[f"genome-level_{path.stem}"] = path
        if group == "text" and "token-level" in LEVELS:
            WEIGHTS[f"token-level_{path.stem}"] = path
        if group == "text" and "sequence-level" in LEVELS:
            WEIGHTS[f"sequence-level_{path.stem}"] = path

SAE_OUT_ALL = []
for pref, outdir in sae_gen.values():
    SAE_OUT_ALL.extend(expand(f"{outdir}/{pref}_{{m}}", m=WEIGHTS.keys()))

# * Rules


rule all:
    input:
        *SAE_OUT_ALL,
        *expand(
            f"{OUTDIRS['sae']}/.{{w}}_{{n}}.csv",
            w=WEIGHTS.keys(),
            n=("concept_scores", "latent_counts"),
        ),
        *expand(
            f"{OUTDIRS['sae']}/{{d}}/{{w}}",
            w=WEIGHTS.keys(),
            d=("latent_umap", "activation_plots"),
        ),
        weights=expand("{o}/weights/{m}.pth", o=OUTDIRS["sae"], m=WEIGHTS.keys()),


rule train_sae:
    input:
        lambda wc: WEIGHTS[f'{wc.get("level")}_{wc.get("dataset")}'],
    output:
        f"{OUTDIRS['sae']}/weights/{{level,[a-z]+-level}}_{{dataset}}.pth",
    params:
        level=lambda wc: wc.get("level"),
        outdir=Path(OUTDIRS["sae"]),
        caches=Path(f"{DPATH}/embedded"),
        pooled=Path(f"{DPATH}/pooled"),
    resources:
        **GPU20,
    log:
        f"{LOGDIR}/interpret/train_sae_{{level}}_{{dataset}}.log",
    script:
        "scripts/interpret.py"


for rname, (prefix, outdir) in sae_gen.items():

    rule:
        name:
            rname
        params:
            caches=Path(f"{DPATH}/embedded"),
            outdir=outdir,
            weight_dict=WEIGHTS,
            pooled=Path(f"{DPATH}/pooled"),
        input:
            f"{OUTDIRS['sae']}/weights/{{weights}}.pth",
        output:
            directory(f"{outdir}/{prefix}_{{weights}}"),
        log:
            f"{LOGDIR}/interpret/{rname}-{{weights}}",
        resources:
            **GPU20,
        script:
            "scripts/interpret.py"


rule eval_sae:
    input:
        f"{sae_gen['save_activations'][1]}/sae-act_{{acts}}",
    output:
        concept_scores=f"{OUTDIRS['sae']}/.{{acts}}_concept_scores.csv",
        latent_counts=f"{OUTDIRS['sae']}/.{{acts}}_latent_counts.csv",
        umap=directory(f"{OUTDIRS['sae']}/latent_umap/{{acts}}"),
        activation_plots=directory(f"{OUTDIRS['sae']}/activation_plots/{{acts}}"),
    params:
        caches=Path(f"{DPATH}/embedded"),
        weight_dict=WEIGHTS,
        outdir=OUTDIRS["sae"],
        pooled=Path(f"{DPATH}/pooled"),
        seqs=Path(f"{DPATH}/processed_sequences"),
    script:
        "scripts/interpret.py"
