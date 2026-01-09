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
    with_metadata,
)
from datasets import Dataset
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from snakemake.script import snakemake as smk
except ImportError:
    smk = type("snakemake", (), {"rule": None, "config": {}, "log": [0]})


logger.enable("amr_predict")
if len(smk.log) == 1:
    logger.add(smk.log[0])

if smk.rule.startswith("train_sae"):
    RCONFIG = smk.config["train_sae"]
else:
    RCONFIG = smk.config.get(smk.rule)

RNG: int = smk.config["rng"]
EMBEDDING = smk.config["embedding"]
X_KEY = smk.config["pool_embeddings"]["key"]
TEST = smk.config["test"]
SEED = smk.config["rng"]
OUTDIR = smk.params.get("outdir")
EMBEDDING_LEVEL: TypeAlias = Literal["sequence-level", "token-level", "genome-level"]
TEXT_KEY = "sequence_aa" if EMBEDDING == "esm" else "sequence"


logger.info(f"cuda available: {torch.cuda.is_available()} (rule {smk.rule})")


def get_level_name(path: Path) -> tuple[EMBEDDING_LEVEL, str]:
    for level in get_args(EMBEDDING_LEVEL):
        if path.stem.startswith(level):
            return level, path.stem.removeprefix(f"{level}_")
    return "genome-level", path.stem


def get_dataset(
    name: str,
    level: EMBEDDING_LEVEL,
    model_name: str,
    key_df: pl.DataFrame | None = None,
) -> Dataset | pl.DataFrame:
    if key_df is None:
        key_df = load_as(smk.params["model_dict"][model_name], "polars")
    if level == "genome-level":
        dset: Dataset = load_as(smk.params["pooled"].joinpath(name)).with_format(
            "torch"
        )
        dset = dset.filter(lambda x: x["sample"] in key_df["sample"])
        return dset
    cache_path = smk.params["caches"].joinpath(f"{name}_{EMBEDDING}_cache")
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
    for dict_path in smk.input:
        level, dset_name = get_level_name(Path(dict_path))
        logger.info(f"Saving for dataset {dset_name}")
        out_name = dset_name if level == "genome-level" else f"{level}_{dset_name}"
        out_name = f"recon_{out_name}" if reconstruct else f"sae-act_{out_name}"
        save_to = smk.params["outdir"] / out_name
        if str(save_to) not in smk.output:
            logger.warning(f"ignoring dataset {save_to}")
            continue
        dataset: Dataset | LinkedDataset = get_dataset(
            dset_name, model_name=Path(dict_path).name, level=level, key_df=None
        )
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
        Dataset.from_dict(into_dset).save_to_disk(save_to)


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
    model_name = f"{level}_{path.stem}.pth"
    dset: Dataset | LinkedDataset = get_dataset(
        name=dset_name, model_name=model_name, level=level, key_df=subset
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
    latent_summary = {"type": [], "dataset": [], "count": [], "activation_source": []}
    concept_scoring = []
    # TODO: refactor this to use the saved activations
    for acts_path in smk.input:
        level: EMBEDDING_LEVEL
        level, dset_name = get_level_name(acts_path)

        concepts: Sequence = RCONFIG["concept_cols"][level]
        meta_cols = ("sample", "ast")
        if level == "sequence-level":
            meta_cols = meta_cols + ("sequence",)

        # Retrieves the full dataset
        for group in ("sae", "model_raw"):
            meta: pl.DataFrame
            if group == "sae":
                dataset: Dataset = load_as(acts_path)
            else:
                dataset = load_as(smk.params["seqs"] / dset_name)
            dataset, meta = with_metadata(dataset, smk.config, meta_cols, align=True)
            SE: EvalSAE = EvalSAE(dataset[X_KEY][:])
            latent_cats: dict = SE.categorize_latents(dense_threshold=1 / 10, save=True)
            for ltype in ["dead", "dense", "sparse"]:
                latent_summary["type"].append(ltype)
                latent_summary["count"].append(len(latent_cats[ltype]))
                latent_summary["dataset"].append(dset_name)
                latent_summary["activation_source"].append(group)

            # NOTE: some preprocessing on concept cols to collapse them into a single
            # label column. would be ideal, but not possible because it would just create
            # so many specific cases. Or is this a good thing?
            topk_plot = RCONFIG["activation_density_topk"]
            SE.drop_latents(drop_dead=True, include_samples=True, inplace=True)
            SE.umap(False, **RCONFIG["umap"])
            latent_clusters: pl.DataFrame = SE.cluster_latents(False)
            for concept in concepts:
                best_latents: pl.DataFrame = (
                    SE.score_latents(labels=meta[concept])
                    .sort("max_activation_prop", descending=True)
                    .with_columns(
                        pl.lit(concept).alias("concept"),
                        pl.lit(dset_name).alias("dataset"),
                        pl.lit(group).alias("activation_source"),
                    )
                    .join(latent_clusters, on="latent_idx")
                )
                concept_scoring.append(best_latents)
                for idx in best_latents["latent_idx"][:topk_plot]:
                    plots: dict = SE.plot_activation_density(idx, meta, [concept])
                    for k, v in plots.items():
                        v: gg.ggplot
                        savepath = (
                            f"{OUTDIR}/activation_plots/{dset_name}/{k}_{group}.png"
                        )
                        v.save(savepath)
                u_savepath = f"{OUTDIR}/latent_umap/{dset_name}/{concept}.png"
                SE.plot_umap(labels=meta[concept]).figure.savefig(u_savepath)
    # TODO: should look into using a concept to guide clustering
    # TODO: add a routine here that checks whether a latent does multiple concepts,
    # can do this after aggregating them all, grouping by idx, dataset etc.

    summary_df = pl.DataFrame(latent_summary)
    summary_plot = (
        gg.ggplot(summary_df, gg.aes(x="type", fill="dataset"))
        + gg.geom_bar(position="dodge")
        + gg.ggtitle(title="Count of latent categories")
    )
    summary_plot.save(smk.output["latent_summary_plot"])
    summary_df.write_csv(smk.output["latent_summary_data"])
    concept_df: pl.DataFrame = pl.concat(concept_scoring)
    concept_df.write_csv(smk.output["concept_scoring_data"])


# * Entry


if fn := globals().get(smk.rule):
    fn()
elif smk.rule.startswith("train_sae"):
    train_sae()
