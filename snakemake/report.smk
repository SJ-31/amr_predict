include: "Snakefile"


INPUTS = {}
RESULTS = {}

INDIR = Path(f"{OUT}/{IN_DATE}")


# * Define output
# ** Model evaluation


def define_eval_out(key, cat):
    eval_tasks = {
        "ctrl_cv": "Cross-validation (control tasks)",
        "cv": "Cross-validation",
    }
    edir = INDIR / key
    INPUTS["evaluation"] = {}
    for group in ("cv", "ctrl_cv", "holdout"):
        if not (edir / group).exists():
            continue
        k = f"{key}_{group}"
        RESULTS[k] = report(
            directory(edir / f".{group}"),
            patterns=["{metric}_{task}.png"],
            category=cat,
            subcategory=eval_tasks.get(group, group),
            labels={"Metric": "{metric}", "Type": "{task}"},
        )
        for task in ("classification", "regression"):
            INPUTS["evaluation"][f"{group}_{task[0]}"] = list(
                (edir / group).rglob(f"*{task}.csv")
            )
    del INPUTS["evaluation"]["ctrl_cv_r"]


# ** Embedding comparison


def define_ec_out(key, cat):
    ec_dir = INDIR / key
    for group in ("pooled", "pooled_distance_correlation", "sequences"):
        if not (ec_dir / group).exists():
            continue
        if group == "pooled_distance_correlation":
            RESULTS[f"{key}_{group}"] = report(
                directory(ec_dir / group),
                patterns=["{dataset}.png"],
                category=cat,
                subcategory=group.replace("_", " ").title(),
                labels={"Dataset": "{dataset}"},
            )
        else:
            for scale in ("continuous", "discrete"):
                pattern = "{dataset}/{iter,\d+}_{dataset}_{ptype,pca|umap}.png"
                labels = {
                    "Iteration": "{iter}",
                    "Dataset": "{dataset}",
                    "Type": "{ptype}",
                }
                RESULTS[f"{key}_{group}_dim_reduction"] = report(
                    directory(ec_dir / group / f"plots-{scale[0]}"),
                    patterns=[pattern],
                    category=cat,
                    subcategory=f"{group.title()} Plots ({scale.title()})",
                    labels=labels,
                )
            RESULTS[f"{key}_{group}_plots"] = report(
                directory(ec_dir / group / ".summary_plots"),
                patterns=["{pname}.png"],
                category=cat,
                subcategory=group.title(),
                labels={"Analysis": "{pname}"},
            )


# ** SAE


def define_sae_out(key, cat):
    sae_dir = INDIR / key
    cat = "Interpretation"
    for pdir in ("latent_umap", "activation_plots"):
        RESULTS[f"{key}_{pdir}"] = report(
            directory(sae_dir / pdir),
            patterns=["{level}-level_{dataset}/{source,model_raw|sae}_{concept}.png"],
            category=cat,
            subcategory=f"SAE {pdir.replace("_", " ").title()}",
            labels={
                "Dataset": "{dataset}",
                "Level": "{level}",
                "Activation Source": "{source}",
                "Concept group": "{concept}",
            },
        )
    RESULTS[f"{key}_latent_counts"] = sae_dir / "latent_counts.csv"
    RESULTS[f"{key}_latent_counts_plot"] = report(
        sae_dir / "latent_fractions_plot.svg",
        category=cat,
        labels={"Name": "SAE Latent Type Composition"},
    )
    RESULTS[f"{key}_concept_score"] = sae_dir / "concept_scores.csv"
    RESULTS[f"{key}_latent_score_plots"] = report(
        directory(sae_dir / "score_plots"),
        patterns=["{concept}.svg"],
        category=cat,
        subcategory="Latent Concept Scoring",
        labels={"Concept": "{concept}"},
    )


# ** Write to RESULTS

for group, fn in {
    "embedding_comparison": define_ec_out,
    "sae": define_sae_out,
    "evaluation": define_eval_out,
}.items():
    if (INDIR / group).exists():
        category = group.replace("_", " ").title()
        fn(group, category)

# * Rules


rule all:
    input:
        **RESULTS,


rule evaluation:
    params:
        **INPUTS["evaluation"],
        outdir=INDIR / "evaluation",
    output:
        directory(f"{INDIR}/evaluation/.{{eval_task}}"),
    script:
        "scripts/report.py"


for level, rname in zip(
    ("pooled", "sequences"), ("compare_pooled", "compare_embeddings")
):

    rule:
        name:
            rname
        input:
            INDIR / "embedding_comparison" / level,
        output:
            directory(INDIR / "embedding_comparison" / level / ".summary_plots"),
        params:
            rule=rname,
        script:
            "scripts/report.py"


rule eval_sae:
    input:
        INDIR / "sae",
    output:
        latent_counts=INDIR / "sae" / "latent_counts.csv",
        concept_scores=INDIR / "sae" / "concept_scores.csv",
        frac_plot=INDIR / "sae" / "latent_fractions_plot.svg",
        score_plot=directory(INDIR / "sae" / "score_plots"),
    script:
        "scripts/report.py"
