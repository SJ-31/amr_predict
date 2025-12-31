#!/usr/bin/env ipython

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeAlias, get_args

import lightning as L
import plotnine as gg
import polars as pl
import torch
from amr_predict.evaluation import (
    categorize_latents,
    highest_activations,
    plot_activation_density,
    score_latents,
)
from amr_predict.sae import BatchTopK
from amr_predict.sae_external import get_default_cfg
from amr_predict.utils import EmbeddingCache, ModuleConfig, load_as, with_metadata
from datasets import Dataset, DatasetDict
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from snakemake.script import snakemake as smk
except ImportError:
    smk = type("snakemake", (), {"rule": None, "config": {}, "log": [0]})

    RCONFIG = smk.config(smk.rule)
    RNG: int = smk.config["rng"]

    logger.enable("amr_predict")
    logger.add(smk.log[0])
    EMBEDDING = smk.config["embedding"]
    X_KEY = smk.config["pool_embeddings"]["key"]
    SEED = smk.config["rng"]


EMBEDDING_LEVEL: TypeAlias = Literal["genome-level", "sequence-level", "token-level"]


def get_dataset(
    name: str,
    level: EMBEDDING_LEVEL,
    key_df: pl.DataFrame | None = None,
    as_df: bool = False,
) -> Dataset | pl.DataFrame:
    if key_df is None:
        seq_dset_name = name if level != "genome-level" else name.split("_")[0]
        key_df = load_as(f"{smk.params["sequences"]}/{seq_dset_name}", "polars")
    if level == "genome-level":
        dset: Dataset = load_as(smk.params["pooled"].joinpath(name))
        dset = dset.filter(lambda x: x["sample"] in key_df["sample"])
        return dset
    cache_path = Path(smk.params["caches"]).joinpath(f"{name}_{EMBEDDING}_cache")
    cache: EmbeddingCache = EmbeddingCache(cache_path)
    dset = cache.to_dataset(
        df=key_df,
        key_col="sequence",
        new_col=X_KEY,
        tokens=level != "sequence-level",
    )
    return dset


def train_routine(
    path: Path, level: EMBEDDING_LEVEL, key_df: pl.DataFrame, outdir: Path
):
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
    dset: Dataset = get_dataset(name=path.stem, level=level, key_df=subset)

    dset_dict: DatasetDict = dset.train_test_split()
    train_kws = RCONFIG["trainer"]
    train_kws.update(run_params.get("trainer", {}))
    train_kws["logger"] = WandbLogger(f"sae_training:{level}", project="amr_predict")
    load_kws = RCONFIG["dataloader"]
    load_kws.update(run_params.get("dataloader", {}))
    trainer = L.Trainer(**train_kws)
    train = DataLoader(dset_dict["train"], **load_kws)
    model: L.LightningModule = get_model_with_defaults(train_dset=train)
    trainer.fit(model, train_dataloaders=train)
    save_path = outdir.joinpath(f"{level}_{path.stem}.pth")
    torch.save(model.state_dict(), save_path)


def get_model_with_defaults(train_dset):
    model_name: str = RCONFIG["model"]
    defaults = get_default_cfg()
    defaults.update(smk.config["models"][model_name])
    if train_dset is not None:
        defaults["act_size"] = next(iter(train_dset))[X_KEY].shape[0]
    cfg = ModuleConfig(**defaults)
    if model_name == "BatchTopK":
        model = BatchTopK(cfg, x_key="embedding")
    return model


# * Rules


def train_sae():
    wanted_cols = ("sample", "seqid")
    for group, paths in smk.input.items():  # Input are either pooled or text datasets
        for level in get_args(EMBEDDING_LEVEL):
            if (not RCONFIG[level]["run"]) or (
                level == "genome-level" and group != "pooled"
            ):
                continue
            cols = wanted_cols.copy()
            if group != "pooled":
                cols += ("sequence",)
            for path in paths:
                keys: pl.Dataframe = load_as(path, "polars").select(cols)
                # Takes processed datasets as input
                train_routine(Path(path), level, keys, Path(smk.params["outdir"]))


def eval_sae():
    latent_summary = {"type": [], "activation_source": [], "count": []}
    for dict_path in smk.input:
        level: EMBEDDING_LEVEL
        level, dset_name = Path(dict_path).stem.split("_", 1)
        model: L.LightningModule = get_model_with_defaults()
        model.load_state_dict(dict_path)
        dataset: Dataset = get_dataset(dset_name, level=level, key_df=None, as_df=True)
        meta: pl.DataFrame
        dataset, meta = with_metadata(dataset, smk.config, ("sample",), align=True)
        # Retrieves the full dataset
        sae_acts: Tensor = model.predict_step(dataset[X_KEY][:])
        # neurons: Tensor = dataset[X_KEY][:]

        latent_cats: dict = categorize_latents(sae_acts)
        for ltype in ["dead", "dense", "sparse"]:
            latent_summary["type"].append(ltype)
            latent_summary["count"].append(len(latent_cats[ltype]))
            latent_summary["activation_source"].append(dset_name)

        concepts: Sequence = RCONFIG["concept_cols"][level]

        # TODO: consider some preprocessing on concept cols to collapse them into a single
        # label column. Otherwise, gonna have to look at overlapping label columns that
        # you need to reconcile...
        topk_plot = RCONFIG["activation_density_topk"]
        for concept in concepts:
            # top_activations: dict = highest_activations(
            #     alive, meta, concept, **RCONFIG["highest_activations"]
            # )
            best_latents: pl.DataFrame = score_latents(
                sae_acts, labels=meta[concept]
            ).sort("max_activation_prop", descending=True)
            best_latents.write_csv(smk.output[""])
            for idx in best_latents["latent_idx"][:topk_plot]:
                plots: dict = plot_activation_density(sae_acts, idx, meta, [concept])
                for k, v in plots.items():
                    v: gg.ggplot
                    savepath = f"{smk.params['outdir']}/activation_plots/{k}.png"
                    v.save(savepath)

    summary_df = pl.DataFrame(latent_summary)
    summary_plot = (
        gg.ggplot(summary_df, gg.aes(x="type", fill="activation_source"))
        + gg.geom_bar(position="dodge")
        + gg.ggtitle(title="Count of latent categories")
    )
    summary_plot.save(smk.output["latent_summary_plot"])
    summary_df.write_csv(smk.output["latent_summary_data"])


# * Entry

if not (fn := globals().get(smk.rule)):
    raise ValueError("Function for rule `{smk.rule}` not defined in this file")
else:
    fn()

# TODO: run the sae metrics on the raw embeddings as well as the sae activations to
# ensure that the latents represent concepts not found in the neurons
