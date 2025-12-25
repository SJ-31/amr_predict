#!/usr/bin/env ipython

import re
from pathlib import Path

import lightning as L
import polars as pl
import torch
from amr_predict.sae import BatchTopK
from amr_predict.sae_external import get_default_cfg
from amr_predict.utils import EmbeddingCache, ModuleConfig, load_as
from datasets import Dataset, DatasetDict
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
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


def train_sae(path: Path, level: str, key_df: pl.DataFrame, outdir: Path):
    run_params: dict = RCONFIG[level]
    chosen_samples: list = (
        key_df.unique("sample")
        .sample(n=run_params["n"], seed=smk.config["rng"])["sample"]
        .to_list()
    )
    if level == "genome-level":
        dset: Dataset = load_as(smk.params["pooled"].joinpath())
        dset = dset.filter(lambda x: x["sample"] in chosen_samples)
    else:
        cache_path = Path(smk.params["caches"]).joinpath(
            f"{path.stem}_{EMBEDDING}_cache"
        )
        cache: EmbeddingCache = EmbeddingCache(cache_path)
        tmp = key_df.filter(pl.col("sample").is_in(chosen_samples))
        subset = pl.concat(
            [df.sample(n=run_params["n_sequence"]) for _, df in tmp.group_by("sample")]
        )
        if level == "sequence-level":
            dset = cache.to_dataset(df=subset, key_col="sequence", new_col=X_KEY)
        else:
            tokens = cache.retrieve(subset["sequence"], tokens=True).rename(
                {"token": X_KEY}
            )
            subset = subset.join(tokens, left_on="sequence", right_on="key").explode(
                X_KEY
            )
            dset = Dataset.from_polars(subset)
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


# Output should be trained SAE for each dataset. Text datasets can be at
# sequence level or the token level. Pooled datasets are just from their name

if smk.rule == "train_sae":
    outdir = Path(smk.params["outdir"])
    wanted_cols = ("sample", "seqid")
    for group, paths in smk.input.items():
        for level in ("token-level", "genome-level", "sequence-level"):
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
                train_sae(Path(path), level, keys, outdir)
if smk.rule == "eval_sae":
    for state_dict in smk.input:
        level = re.sub("_.*", "", Path(state_dict).stem)
        model: L.LightningModule = get_model_with_defaults()
        model.load_state_dict(state_dict)
        # TODO: you still need activations from the dataset, so wrap the dataset prep
        # routine up in a function
        # dset =


# TODO: run the sae metrics on the raw embeddings as well as the sae activations to
# ensure that the latents represent concepts not found in the neurons
