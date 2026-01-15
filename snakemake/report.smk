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
        if group == "holdout" and not config["holdout"]["splits"]:
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
            INPUTS["evaluation"][f"{group}_{task[0]}"] = (edir / group).rglob(
                f"*{task}.csv"
            )


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
            # for ptype in ("umap", "pca"):
            #     RESULTS[f"{key}_{group}_{ptype}"] = report(
            #         directory(ec_dir / group),
            #         patterns=[f"{{dataset}}_plots/{{iter}}_{{dataset}}_{ptype}.png"],
            #         category=cat,
            #         subcategory=f"{group} plots ({group})",
            #         labels={
            #             "Iteration": "{iter}",
            #             "Dataset": "{dataset}",
            #             "Type": "{type}",
            #         },
            #     )
            RESULTS[f"{key}_{group}_plots"] = report(
                directory(ec_dir / group / ".summary_plots"),
                patterns=["{pname}.png"],
                category=cat,
                subcategory=group.title(),
                labels={"Analysis": "{pname}"},
            )


# ** SAE


def define_sae_out(key, cat):
    return


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
    input:
        INPUTS["evaluation"],
    params:
        outdir=INDIR / "evaluation",
    output:
        f"{INDIR}/evaluation/.{{eval_task}}",
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
