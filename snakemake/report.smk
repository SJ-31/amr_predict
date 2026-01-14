include: "Snakefile"


RESULTS = {}

INDIR = Path(f"{OUT}/{IN_DATE}")


# * Figure definitions
# ** Model evaluation
EVAL_TASKS = {
    "ctrl_cv": "Cross-validation (control tasks)",
    "cv": "Cross-validation",
}


def eval_figures(cat):
    outdir = INDIR / "evaluation"
    result = {}
    file_mapping = {}
    for group in ("cv", "ctrl_cv", "holdout"):
        if group == "holdout" and not config["holdout"]["splits"]:
            continue
        key = f"{group}_{task}"
        result[key] = report(
            directory(OUTDIR / f".{group}"),
            patterns=["{metric}_{task}.png"],
            category=cat,
            subcategory=EVAL_TASKS.get(group, group),
            labels={"Metric": "{metric}", "Type": "{task}"},
        )

    return result


# ** Embedding comparison
def comparison_figures(cat):
    path = INDIR / "embedding_comparison"
    result = {}
    for group in ("pooled", "pooled_distance_correlation", "sequences"):
        if group == "pooled_distance_correlation":
            result[group] = report(
                directory(INDIR / group),
                patterns=["{dataset}.png"],
                category=cat,
                subcategory=group.replace("_", " ").title(),
                labels={"Dataset": "{dataset}"},
            )
        # For pooled and sequences, you want the PCAs and um
        elif group == "pooled":
            result[group] = report()
            # TDO


# * Rules and output

# TODO: make it flat
for group in ("embedding_comparison", "sae", "evaluation"):
    if (DIR / group).exists():
        if group == "evaluation":
            RESULTS[f"{group}_figures"] = eval_results()
            RESULTS[f"{group}_tables"] = TODO  # datavzrd stuff here
        elif group == "sae":
            pass


rule all:
    input:
        **RESULTS,


rule evaluation:
    input:
        DIR / "evaluation",
