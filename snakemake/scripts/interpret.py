#!/usr/bin/env ipython

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias, get_args

import lightning as L
import plotnine as gg
import polars as pl
import torch
from amr_predict.evaluation import EvalSAE
from amr_predict.sae import BatchTopK
from amr_predict.sae_external import get_default_cfg
from amr_predict.utils import (
    EmbeddingCache,
    LinkedDataset,
    ModuleConfig,
    load_as,
    plot_params,
    with_metadata,
)
from datasets import Dataset
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float32)

try:
    from snakemake.script import snakemake as smk
except ImportError:
    smk = type("snakemake", (), {"rule": None, "config": {}, "log": [0]})


CONFIG = smk.config

logger.enable("amr_predict")
if len(smk.log) == 1:
    logger.add(smk.log[0])

if smk.rule.startswith("train_sae"):
    RCONFIG = smk.config["train_sae"]
else:
    RCONFIG = smk.config.get(smk.rule)

EMBEDDING = smk.config["embedding"]
X_KEY = smk.config["pool_embeddings"]["key"]
TEST = smk.config["test"]
SEED = smk.config["rng"]
EMBEDDING_LEVEL: TypeAlias = Literal["sequence-level", "token-level", "genome-level"]
TEXT_KEY = "sequence_aa" if EMBEDDING == "esm" else "sequence"


logger.info(f"cuda available: {torch.cuda.is_available()} (rule {smk.rule})")


def get_level_name(path: Path) -> tuple[EMBEDDING_LEVEL, str]:
    cleaned = path.stem.removeprefix("sae-act_").removeprefix("recon_")
    for level in get_args(EMBEDDING_LEVEL):
        if cleaned.startswith(f"{level}_"):
            cleaned = cleaned.removeprefix(f"{level}_")
            if level != "genome-level":
                cleaned = cleaned.removesuffix(f"-{EMBEDDING}")
            return level, cleaned
    return "genome-level", cleaned.removeprefix("genome-level_")


def maybe_subsample(
    data: Dataset | LinkedDataset, level: EMBEDDING_LEVEL
) -> LinkedDataset | Dataset:
    is_linked = isinstance(data, LinkedDataset)
    cols = ["sample"]
    if level == "sequence-level":
        cols.append(TEXT_KEY)
    tmp = data.select_columns(cols)
    df: pl.DataFrame = tmp.meta if is_linked else tmp.to_polars()
    if (n := RCONFIG[level].get("n")) and level == "genome-level":
        chosen = df.with_row_index().sample(
            n, seed=SEED, with_replacement=False, shuffle=True
        )["index"]
        return data.select(chosen.to_list())
    elif n and level == "sequence-level":
        chosen_samples = df.unique("sample").sample(
            n, seed=SEED, with_replacement=False, shuffle=True
        )["sample"]
        if is_linked:
            data = data.filter(lambda x: x["sample"].is_in(chosen_samples))
        else:
            data = data.filter(lambda x: x["sample"] in chosen_samples)
        df = df.filter(pl.col("sample").is_in(chosen_samples))
    if n_s := RCONFIG[level].get("n_sequence"):
        idx = []
        df = df.with_row_index()
        for _, gdf in df.group_by("sample"):
            if gdf.height > n_s:
                gdf = gdf.sample(n_s, seed=SEED, shuffle=True, with_replacement=False)
            idx.extend(gdf["index"].to_list())
        data = data.select(idx)
    return data


def get_dataset(
    name: str,
    level: EMBEDDING_LEVEL,
    weights_name: str | None = None,
    key_df: pl.DataFrame | None = None,
) -> Dataset | pl.DataFrame:
    if key_df is None:
        key_df = load_as(smk.params["weight_dict"][weights_name], "polars")
    if level == "genome-level":
        dset: Dataset = load_as(
            smk.params["pooled"].joinpath(name), "huggingface"
        ).with_format("torch", dtype=torch.float32)
        dset = dset.filter(lambda x: x["sample"] in key_df["sample"])
        return dset
    cache_path = smk.params["caches"].joinpath(f"{name}-{EMBEDDING}_cache")
    cache: EmbeddingCache = EmbeddingCache(cache_path)
    dset = cache.to_dataset(
        df=key_df,
        key_col=TEXT_KEY,
        new_col=X_KEY,
        tokens=level != "sequence-level",
    )
    return dset


def get_model_with_defaults(train_dset: Dataset | Any):
    config = smk.config["train_sae"]
    model_name: str = config["model"]
    defaults = get_default_cfg()
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to cpu")
        defaults["device"] = "cpu"
    defaults.update(smk.config["models"][model_name]["kws"] or {})
    if train_dset is not None:
        if isinstance(train_dset, Dataset):
            defaults["act_size"] = len(next(iter(train_dset))[X_KEY])
        else:
            defaults["act_size"] = next(iter(train_dset))[X_KEY].shape[1]
        defaults["dict_size"] = defaults["act_size"] * config["expansion_factor"]
    cfg = ModuleConfig(**defaults)
    if model_name == "BatchTopK":
        model = BatchTopK(cfg, x_key=X_KEY)
    else:
        raise ValueError("Model name not recognized")
    return model


def save_from_sae(reconstruct: bool = False):
    dict_path = smk.input[0]
    level, dset_name = get_level_name(Path(dict_path))
    logger.info(f"Saving for dataset {dset_name}")
    dataset: Dataset | LinkedDataset = get_dataset(
        dset_name, weights_name=Path(dict_path).stem, level=level, key_df=None
    )
    logger.info("{} {}, n before subsampling: {}", level, dset_name, dataset.shape[0])
    dataset = maybe_subsample(dataset, level)
    logger.info("{} {}, n after subsampling: {}", level, dset_name, dataset.shape[0])
    model: L.LightningModule = get_model_with_defaults(dataset)
    model.load_state_dict(torch.load(dict_path, weights_only=True))
    model.to(model.device)
    x = dataset[X_KEY][:].to(model.device)
    if reconstruct:
        vals: Tensor = model.reconstruct(x)
        logger.success("Reconstruction complete")
    else:
        vals = model.predict_step(x)
        logger.success("Generating activations complete")
    into_dset = {X_KEY: vals, "sample": dataset["sample"][:]}
    if level == "sequence-level":
        into_dset["uid"] = dataset["uid"][:]
        # TODO: should probably add a condition for token-level, but you probably
        # won't be using it
    Dataset.from_dict(into_dset).save_to_disk(smk.output[0])


# * Rules


def train_sae():
    level: EMBEDDING_LEVEL = smk.params["level"]
    path: Path = Path(smk.input[0])
    wanted_cols = ("sample",)
    cols = wanted_cols + (TEXT_KEY,) if level != "genome-level" else wanted_cols
    key_df = load_as(path, "polars").select(cols)
    run_params: dict = RCONFIG[level]
    # Don't use all samples for SAE training
    chosen_samples: list = (
        key_df.unique("sample").sample(n=run_params["n"], seed=SEED)["sample"].to_list()
    )
    if level == "genome-level":
        subset = key_df.filter(pl.col("sample").is_in(chosen_samples))
    else:
        tmp = key_df.filter(pl.col("sample").is_in(chosen_samples))
        subset = pl.concat(
            [
                df.sample(n=run_params["n_sequence"], seed=SEED)
                for _, df in tmp.group_by("sample")
            ]
        )
    dset_name = path.stem
    model_name = f"{level}_{path.stem}"
    dset: Dataset | LinkedDataset = get_dataset(
        name=dset_name, weights_name=model_name, level=level, key_df=subset
    )

    # TODO: review how to train saes
    # Fairly certain you don't need to do train-test splits when training SAEs
    # dset_dict: DatasetDict = DatasetDict()
    # train_idx, test_idx = train_test_split(range(dset.shape[0]), **RCONFIG["splits"])
    # dset_dict["train"] = dset.select(train_idx)
    # dset_dict["test"] = dset.select(test_idx)
    load_kws = RCONFIG["dataloader"]
    load_kws.update(run_params.get("dataloader", {}) or {})
    # train = DataLoader(dset_dict["train"], **load_kws)
    # test = DataLoader(dset_dict["test"], **load_kws)
    test = None

    train_kws = RCONFIG["trainer"]
    train_kws.update(run_params.get("trainer", {}) or {})
    run_name = f"train_sae:{level}_{dset_name}{"-test" if TEST else ""}"
    if smk.config["log_wandb"]:
        train_kws["logger"] = WandbLogger(run_name, project="amr_predict")
    trainer = L.Trainer(**train_kws)
    train = DataLoader(dset, **load_kws)

    model: L.LightningModule = get_model_with_defaults(train_dset=train)
    logger.info("Training started")
    trainer.fit(model, train_dataloaders=train)
    logger.success("Training complete")
    torch.save(model.state_dict(), smk.output[0])


def reconstruct_datasets():
    save_from_sae(True)


def save_activations():
    save_from_sae(False)


def eval_sae():
    latent_summary = {"type": [], "count": [], "activation_source": []}
    concept_scoring = []
    acts_path = Path(smk.input[0])
    level: EMBEDDING_LEVEL
    level, dset_name = get_level_name(acts_path)
    dset_name = (
        dset_name
        if level == "genome-level"
        else dset_name.removesuffix(f"-{EMBEDDING}")
    )
    concepts: Sequence = RCONFIG["concept_cols"][level]
    meta_cols = ("sample", "ast")
    if level == "sequence-level":
        meta_cols = meta_cols + ("sequence",)

    # Retrieves the full dataset
    for group in ("sae", "model_raw"):
        meta: pl.DataFrame
        if group == "sae":
            dataset: Dataset = load_as(acts_path)
        elif group == "model_raw" and level != "genome-level":
            dataset = get_dataset(
                dset_name,
                level=level,
                key_df=load_as(smk.params["seqs"] / dset_name, "polars"),
            )
        elif group == "model_raw":
            dataset = load_as(smk.params["pooled"] / dset_name)
        else:
            raise ValueError("loading for tokens not implemented yet")
        dataset, meta = with_metadata(
            dataset,
            smk.config,
            "sample",
            meta_cols,
            align=True,
            dset_name=dset_name,
        )
        meta = meta.with_columns(pl.col(concepts).fill_null("unknown"))
        SE: EvalSAE = EvalSAE(dataset[X_KEY][:])
        logger.info("Beginning categorize_latents")
        latent_cats: dict = SE.categorize_latents(dense_threshold=1 / 10, save=True)
        logger.success("categorize_latents complete")
        for ltype in ["dead", "dense", "sparse"]:
            latent_summary["type"].append(ltype)
            latent_summary["count"].append(len(latent_cats[ltype]))
            latent_summary["activation_source"].append(group)

        umap_outdir = Path(smk.output["umap"])
        umap_outdir.mkdir(exist_ok=True)
        act_outdir = Path(smk.output["activation_plots"])
        act_outdir.mkdir(exist_ok=True)
        # NOTE: some preprocessing on concept cols to collapse them into a single
        # label column. would be ideal, but not possible because it would just create
        # so many specific cases. Or is this a good thing?
        topk_plot = RCONFIG["activation_density"]
        SE.drop_latents(drop_dead=True, inplace=True)
        logger.info(f"shape of latent df {SE.acts.shape}")
        logger.info("Starting umap")
        umap_points = pl.DataFrame(
            SE.umap(False, return_array=True, **RCONFIG["umap"])
        ).rename(lambda x: x.replace("column_", "UMAP"))
        umap_points = pl.concat([umap_points, meta], how="horizontal")
        umap_points.write_parquet(umap_outdir / "umap_data.parquet")
        logger.success("umap complete")
        logger.info("Starting latent clustering")
        latent_clusters: pl.DataFrame = SE.cluster_latents(False)
        logger.success("Clustering complete")
        for concept in concepts:
            logger.info(f"Scoring latent with {concept} labels")
            best_latents: pl.DataFrame = (
                SE.score_latents(labels=meta[concept])
                .with_columns(
                    pl.lit(concept).alias("concept"),
                    pl.lit(group).alias("activation_source"),
                )
                .join(latent_clusters, on="latent_idx")
                .join(  # Align for UMAP plot
                    pl.DataFrame({"latent_idx": SE.acts.columns}),
                    on="latent_idx",
                    how="right",
                    maintain_order="right",
                )
            )
            concept_scoring.append(best_latents)
            top_best = best_latents.sort("max_activation_prop", descending=True)[
                "latent_idx"
            ][: topk_plot["topk"]]
            plot: gg.ggplot
            plot, raw = SE.plot_activation_density(
                top_best, meta, [concept], top_labels=None, **topk_plot["kws"]
            )
            raw.write_parquet(act_outdir / f"{group}_{concept}.parquet")
            plot.save(
                act_outdir / f"{group}_{concept}.png",
                **plot_params("sae_activations", CONFIG),
            )
            labels = best_latents["label_max"]
            logger.info(f"Number of labels: {len(meta[concept])}")
            SE.plot_umap(labels=labels).figure.savefig(
                umap_outdir / f"{group}_{concept}.png",
                **plot_params("sae_umap", CONFIG),
            )
        logger.success("Scoring complete")
    # TODO: should look into using a factor to guide clustering
    # TODO: add a routine here that checks whether a latent does multiple factors,
    # can do this after aggregating them all, grouping by idx, dataset etc.
    # TODO: another option is to unpivot the data on the factors, but
    # this is gonna be a huge matrix multiplication
    summary_df = pl.DataFrame(latent_summary).with_columns(
        pl.lit(level).alias("level"), pl.lit(dset_name).alias("dataset")
    )
    summary_df.write_csv(smk.output["latent_counts"])
    concept_df: pl.DataFrame = pl.concat(
        concept_scoring, how="vertical_relaxed"
    ).with_columns(
        pl.lit(dset_name).alias("dataset"),
        pl.lit(level).alias("level"),
    )
    concept_df.write_csv(smk.output["concept_scores"])


# * Entry


if fn := globals().get(smk.rule):
    fn()
elif smk.rule.startswith("train_sae"):
    train_sae()
