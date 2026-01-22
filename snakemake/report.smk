include: "Snakefile"


import shutil


configfile: "models.yaml"


# TODO: write this up, should be the last thing you do
# report: "report/main.rst"


INPUTS = {}
RESULTS = {}
COPY_EXISTING = {}
INDIR = Path(f"{OUT}/{IN_DATE}")


# * Define output


EXISTING_SAVED = INDIR / ".plots_for_report"
ENV_SAVED = INDIR / ".config_for_report"

# ** Model evaluation

EVAL_OUT = {}


def define_eval_out(key, cat):
    eval_tasks = {
        "ctrl_cv": "Cross-validation (control tasks)",
        "cv": "Cross-validation",
        "holdout": "Holdout",
    }
    edir = INDIR / key
    INPUTS["evaluation"] = {}
    for group in ("cv", "ctrl_cv", "holdout"):
        if not (edir / group).exists():
            continue
        k = f"{key}_{group}"
        RESULTS[k] = edir / f".{group}"
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
            COPY_EXISTING[ec_dir / group] = EXISTING_SAVED / group
            RESULTS[f"exists_{key}_{group}"] = report(
                directory(EXISTING_SAVED / group),
                patterns=["{dataset}.png"],
                category=cat,
                subcategory=group.replace("_", " ").title(),
                labels={"Dataset": "{dataset}"},
            )
        else:
            for scale in ("continuous", "discrete"):
                pattern = "{dataset}/{iter,\\d+}_{dataset}_{ptype,pca|umap}.png"
                labels = {
                    "Iteration": "{iter}",
                    "Dataset": "{dataset}",
                    "Type": "{ptype}",
                }
                COPY_EXISTING[ec_dir / group / f"plots-{scale[0]}"] = (
                    EXISTING_SAVED / group / f"plots-{scale[0]}"
                )
                RESULTS[f"exists_{key}_{group}_dim_reduction"] = report(
                    directory(EXISTING_SAVED / group / f"plots-{scale[0]}"),
                    patterns=[pattern],
                    category=cat,
                    subcategory=f"{group.title()} Plots ({scale.title()})",
                    labels=labels,
                )
            RESULTS[f"{key}_{group}_plots"] = ec_dir / group / ".summary_plots"


# ** SAE


def define_sae_out(key, cat):
    sae_dir = INDIR / key
    cat = "Interpretation"
    for pdir in ("latent_umap", "activation_plots"):
        COPY_EXISTING[sae_dir / pdir] = EXISTING_SAVED / pdir
        RESULTS[f"exists_{key}_{pdir}"] = report(
            directory(EXISTING_SAVED / pdir),
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
    RESULTS[f"{key}_latent_score_plots"] = sae_dir / "score_plots"


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
        *expand(f"{INDIR}/.samples_with_{{f}}.csv", f=("ast", "meta")),
        *expand(f"{INDIR}/.datavzrd_{{f}}", f=("seqs", "samples")),
        **{k: str(v) for k, v in RESULTS.items()},
        env_record=ENV_SAVED,
        env_record_files=expand(
            f"{ENV_SAVED}/{{f}}.{{e}}",
            f=(
                "embedding_comparison",
                "embedding_parameters",
                "data_preparation",
                "evaluation",
            ),
            e=("html", "yaml"),
        ),


rule record_env:
    output:
        report(
            directory(ENV_SAVED),
            patterns=["{group}.{file}"],
            category="Configuration",
            labels=lambda wc: {
                "Config group": wc.get("group").replace("_", " ").title(),
                "Mode": {"html": "Display", "yaml": "Yaml (Download)"}[wc.get("file")],
            },
        ),
        *rules.all.input.env_record_files,
    params:
        outdir=ENV_SAVED,
    script:
        "scripts/report.py"


rule gather_existing:
    output:
        **{k: v for k, v in RESULTS.items() if k.startswith("exists")},
    run:
        for src, dest in COPY_EXISTING.items():
            if not dest.exists():
                shutil.copytree(src, dest)


def get_eval_subcategory(wc):
    task = wc.get("eval_task")
    mapping = {
        "ctrl_cv": "Cross-validation (Control tasks)",
        "cv": "Cross-validation",
    }
    if task in mapping:
        return mapping[task]
    return f"Holdout: {wc.get('test_set')}"


rule evaluation:
    params:
        **INPUTS["evaluation"],
        outdir=INDIR / "evaluation",
    output:
        report(
            directory(f"{INDIR}/evaluation/.{{eval_task}}"),
            patterns=["{metric}_{task}.png"],
            category="Evaluation",
            subcategory=lambda wc: eval_tasks.get(
                wc.get("eval_task"), wc.get("eval_task")
            ),
            labels={"Metric": "{metric}", "Type": "{task}"},
        ),
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
            report(
                directory(INDIR / "embedding_comparison" / level / ".summary_plots"),
                patterns=["{pname}.svg"],
                category="Embedding Comparison",
                subcategory=level.title(),
                labels={"Analysis": "{pname}"},
            ),
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
        score_plot=report(
            directory(INDIR / "sae" / "score_plots"),
            patterns=["{concept}.svg"],
            category="Interpretation",
            subcategory="Latent Concept Scoring",
            labels={"Concept": "{concept}"},
        ),
    script:
        "scripts/report.py"


# TODO: [2026-01-16 Fri] add this to a top-level category called Dataset
# use plotly
rule format_metadata:
    input:
        next((INDIR / "datasets" / "pooled").iterdir()),
        INDIR / "seq_metadata.csv",
    output:
        ast=INDIR / ".samples_with_ast.csv",
        meta=INDIR / ".samples_with_meta.csv",
        count_tables=directory(INDIR / ".seq_meta/count_tables"),
        seq_values_per=INDIR / ".seq_meta/values_per_sample.csv",
    script:
        "scripts/report.py"


module shared:
    snakefile:
        "shared.smk"


use rule datavzrd from shared as show_sequence_annotations with:
    input:
        count_dir=rules.format_metadata.output.count_tables,
    params:
        template="report/seq_anno.yaml",
    output:
        dir=report(
            directory(f"{INDIR}/.datavzrd_seqs"),
            category="Dataset",
            htmlindex="index.html",
            labels={"Name": "Sequence annotation counts"},
        ),
        config=f"{INDIR}/evaluation/.datavzrd_seqs.yaml",


use rule datavzrd from shared as show_sample_metadata with:
    input:
        sample_meta=rules.format_metadata.output.meta,
        ast=rules.format_metadata.output.ast,
        seq_meta=rules.format_metadata.output.seq_values_per,
    params:
        template="report/dataset.yaml",
        to_remove=["ProjectID", "BioProject", "TaxID"],
        ast_boolean=[
            t.replace("_class", "_resistant")
            for t in config["tasks"]["classification"]
        ],
        ast=config["tasks"]["regression"],
        source_cols=[
            "collection_year",
            "title",
            "location",
            "sequenced_by",
            "umbrella_project",
            "isolation_source",
            "isolation_source_broad",
        ],
        patho_cols=["strain", "serovar", "serotype", "interest_group"],
        handle_cols=[
            "sequenced_by",
            "laboratory_typing_platform",
            "laboratory_typing_method_version_or_reagent",
            "Platform",
        ],
        tax_cols=[
            "species",
            "subspecies",
            "genus",
            "family",
            "order",
        ],
    output:
        dir=report(
            directory(f"{INDIR}/.datavzrd_samples"),
            category="Dataset",
            htmlindex="index.html",
            labels={"Name": "Sample metadata"},
        ),
        config=f"{INDIR}/evaluation/.datavzrd_samples.yaml",
