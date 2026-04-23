#!/usr/bin/env ipython
import anndata as ad
import polars as pl
from amr_predict.cache import LinkedDataset
from amr_predict.utils import smoothen_log2
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

DSET_TYPES: TypeAlias = Dataset | LinkedDataset | ad.AnnData | pl.DataFrame


def with_metadata(
    dset: DSET_TYPES,
    cfg: dict,
    sample_col: str = "sample",
    meta_options: str | tuple[str, ...] = ("ast", "sample", "sequence", "external"),
    align: bool = False,
    dset_name: str | None = None,
) -> DSET_TYPES | tuple[DSET_TYPES, pl.DataFrame]:
    for_test: bool = cfg["test"]
    if isinstance(dset, ad.AnnData):
        merging: pl.DataFrame = pl.from_pandas(dset.obs)
    elif isinstance(dset, pl.DataFrame):
        merging = dset
    else:
        to_df = {sample_col: dset[sample_col][:]}
        if "sequence" in meta_options:
            to_df["uid"] = dset["uid"][:]
        merging = pl.DataFrame(to_df)
    if isinstance(meta_options, str):
        meta_options = [meta_options]
    for m in meta_options:
        if m == "sequence" and not dset_name:
            raise ValueError(
                "dataset name must be provided if requesting sequence metadata"
            )
        elif m == "sequence":
            start = cfg["out"]["tests"] if for_test else cfg["remote"]
            path = f"{start}/{cfg['in_date']}/datasets/processed_sequences/{dset_name}"
            df = load_as(path, "polars").with_columns(
                pl.any_horizontal(cs.contains("gene").is_not_null()).alias("in_gene")
            )
            key_col = "uid"
        elif m in {"ast", "sample"}:
            df = (
                get_ast_meta(cfg)
                if m == "ast"
                else read_tabular(cfg["sample_metadata"]["file"])
            )
            key_col = cfg[f"{m}_metadata"]["id_col"]
        elif m == "external":
            df = with_external_amr_predictions(
                merging, cfg, sample_col=sample_col, min_args=1
            )
            key_col = sample_col
        else:
            raise ValueError(
                f"metadata type `{m}` must be one of 'ast', 'sequence', 'sample', `external`"
            )
        merging = merging.join(
            df, left_on=sample_col, right_on=key_col, how="left", maintain_order="left"
        )
        if for_test:
            merging = modify_metadata_test(m, df, key_col, merging)
        tmp = (merging.null_count() / merging.height).unpivot()
        null_dict = dict(zip(tmp["variable"], tmp["value"]))
        # logger.info(
        #     "Percentage of nulls in merged metadata\n{}",
        #     null_dict,
        # )
    if not align and isinstance(dset, ad.AnnData):
        new = dset.copy()
        new.obs = merging.to_pandas()
        return new
    elif not align and isinstance(dset, pl.DataFrame):
        return merging
    elif not align:
        to_merge = Dataset.from_polars(merging.drop(sample_col))
        return concatenate_datasets([dset, to_merge], axis=1)
    return dset, merging


def with_external_amr_predictions(
    df: pl.DataFrame, cfg: dict, sample_col: str = "sample", min_args: int = 1
) -> pl.DataFrame:
    """Add confusion matrix codes for predictions made by external AMR tools
    by cross-referencing with observed AST data

    Parameters
    ----------
    cfg : dict
        Dictionary as used in Snakemake pipeline
    min_args : int
        The miniumum number of resistance genes in the genome
        predicted by the tool to consider the tool as calling the genome to have an
        AMR phenotype

    Returns
    -------
    DataFrame where each sample is labeled as TP, TN, FP, FN for each tool in columns
    When possible, resistance predictions are checked with specific antibiotics e.g.
    `TOOL_tetracycline` is checked against the observed resistance to tetracycline.
    `TOOL_any` checks against the observed resistance to any antibiotic

    Notes
    -----
    The external tools are assumed to be aggregated by hAMRonization
    Label key:
    + False negative: no prediction but are resistant
    + False positive: predicted AMR genes but not resistant
    + True positive: predicted AMR genes and resistant
    + True negative: no prediction, not resistant

    TODO: what's a good value to set for min_args?
    TODO: you currently ignore the broad antibiotic classes in the hAMRonization tools,
        placing them all under `other`. Could instead try to classify the antibiotics
        in the AST data and then check against the tool prediction for that class
        e.g. `aminoglycoside antibiotic` would include streptomycin, gentamicin
        and amikacin
        See `broad_class_mapping` and finish it up
    WARNING: the cm codes assume that the AST data are ground truth
    """
    ast_data = get_ast_meta(cfg)
    broad_class_mapping = {"aminoglycoside": ("gentamicin", "amikacin", "streptomycin")}
    known_drugs = [
        col
        for col in ast_data.columns
        if not col.endswith("_class") and col not in {"BioSample", "Run"}
    ]
    # known_drugs.extend(broad_class_mapping.keys())
    df = df.select(sample_col).join(
        ast_data, how="left", left_on=sample_col, right_on=cfg["ast_metadata"]["id_col"]
    )
    hamr = read_format_hamr(cfg, known_drugs, sample_col=sample_col)
    merged = df.join(hamr, on=sample_col, how="left")
    res_val = 1 if cfg["ast_metadata"].get("binarize") else "resistant"
    for col in hamr.columns:
        drug = col.split("_")[1] if "_" in col else ""
        pass_thresh = pl.col(col) >= min_args

        def code_cm(df: pl.DataFrame, expr) -> pl.DataFrame:
            return df.with_columns(
                pl.when(pass_thresh & expr)
                .then(pl.lit("TP"))
                .when(pass_thresh)
                .then(pl.lit("FP"))
                .when(expr)
                .then(pl.lit("FN"))
                .otherwise(pl.lit("TN"))
                .alias(f"{col}_cm")
            )

        if col == sample_col:
            continue
        elif col.endswith("_any"):
            merged = code_cm(merged, pl.col("any_resistant"))
        elif drug_cols := broad_class_mapping.get(drug):
            any_expr = pl.any_horizontal(cs.by_name(drug_cols) == res_val)
            merged = code_cm(merged, any_expr)
        elif f"{drug}_class" in merged.columns:
            merged = code_cm(merged, pl.col(f"{drug}_class") == res_val)
    return merged


def encode_strs(
    data: Dataset | LinkedDataset | ad.AnnData, task_names: tuple
) -> tuple[Dataset | ad.AnnData, dict[str, LabelEncoder]]:
    encoders = {}
    for task in task_names:
        encoder = LabelEncoder()
        if isinstance(data, Dataset):
            task_vec = data[task][:]
            data = data.remove_columns(task).add_column(
                task, encoder.fit_transform(task_vec)
            )
        elif isinstance(data, LinkedDataset):
            data.meta = data.meta.with_columns(
                pl.Series(encoder.fit_transform(task_vec)).alias(task)
            )
        else:
            task_vec = data.obs[task]
            data.obs.loc[:, task] = encoder.fit_transform(task_vec)
        encoders[task] = encoder
    return data, encoders


def get_ast_meta(cfg: dict) -> pl.DataFrame:
    args = cfg["ast_metadata"]
    df = read_tabular(args["file"])
    if not args.get("binarize"):
        df = df.with_columns(
            pl.any_horizontal(cs.ends_with("_class") == "resistant").alias(
                "any_resistant"
            ),
            pl.any_horizontal(cs.ends_with("_class") == "susceptible").alias(
                "any_susceptible"
            ),
        )
    else:
        df = df.with_columns(
            cs.ends_with("_class")
            .replace_strict({"resistant": 1}, default=0)
            .cast(pl.UInt32)
        ).with_columns(
            pl.any_horizontal(cs.ends_with("_class") == 1).alias("any_resistant")
        )
    if args.get("smooth"):
        for col in df.select(cs.by_dtype(pl.Float64)).columns:
            smoothed = smoothen_log2(df[col].to_numpy())
            df = df.with_columns(pl.Series(smoothed).alias(col))
    return df


def modify_metadata_test(
    meta_type: Literal["ast", "sample", "sequence"],
    original_df: pl.DataFrame,
    key_col: str,
    merging: pl.DataFrame,
) -> pl.DataFrame:
    if meta_type == "ast":
        cols = [
            col
            for col, dtype in original_df.schema.items()
            if not isinstance(dtype, pl.Boolean)
        ]
        class_cols = list(filter(lambda x: x.endswith("_class"), cols))
        other_cols = list(
            filter(lambda x: x != key_col and not x.endswith("_class"), cols)
        )
        if original_df.schema[class_cols[0]] == pl.String:
            merging = add_random_cols(
                merging, class_cols, ["resistant", "susceptible", "intermediate"]
            )
        else:
            merging = add_random_cols(merging, class_cols, [0, 1])
        merging = add_random_cols(merging, other_cols, low=0.01, high=1024)
    elif meta_type == "sequence":
        merging = add_random_cols(
            merging,
            cols=filter(lambda x: "gene" in x and x != "in_gene", original_df.columns),
            choices=list(ascii_uppercase)[:10],
        )
    else:
        merging = add_random_cols(
            merging,
            filter(lambda x: x != key_col, original_df.columns),
            choices=list(ascii_uppercase)[:15],
        )
    return merging


def read_format_hamr(
    cfg: dict, known_drugs: Sequence, sample_col: str = "sample"
) -> pl.DataFrame:
    hamr = pl.read_csv(
        cfg["seq_metadata"]["hamronization"],
        separator="\t",
        raise_if_empty=False,
        infer_schema_length=None,
    ).rename({"input_file_name": sample_col})
    to_remove = (
        "\\.mapping.*\\.deeparg",
        "\\.tsv\\.amrfinderplus",
        "\\.txt\\.rgi",
        "_retrieved-genes-.*",
    )
    wanted_cols = (sample_col, "analysis_software_name", "drug_class")
    drug_replacements = {"tobramcyin": "tobramycin", None: None}
    for pat in to_remove:
        hamr = hamr.with_columns(pl.col(sample_col).str.replace(pat, value=""))
    software = hamr["analysis_software_name"].unique()
    fmt = (
        hamr.select(wanted_cols)
        .with_columns(pl.col("drug_class").str.split(";"))
        .explode("drug_class")
        .with_columns(
            pl.col("drug_class")
            .str.to_lowercase()
            .str.strip_chars()
            .replace(drug_replacements)
        )
        .with_columns(
            pl.when(pl.col("drug_class").is_in(known_drugs))
            .then("drug_class")
            .when(pl.col("drug_class").is_null())
            .then(pl.lit(None))
            .otherwise(pl.lit("other_drug"))
        )
        .filter(pl.col("drug_class").is_not_null())
        .with_columns(pl.lit(1).alias("value"))
        .pivot(
            values="value",
            on=["analysis_software_name", "drug_class"],
            aggregate_function="sum",
        )
        .rename(
            lambda x: "_".join(
                x.removeprefix("{").removesuffix("}").replace('"', "").split(",")
            )
            if x.startswith("{")
            else x
        )
        .group_by(sample_col)
        .agg(pl.all().sum())
        .with_columns(
            *[
                pl.sum_horizontal(cs.starts_with(f"{sn}_")).alias(f"{sn}_any")
                for sn in software
            ]
        )
    )
    return fmt
