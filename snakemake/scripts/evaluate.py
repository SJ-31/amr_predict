#!/usr/bin/env ipython

import os
import re
from pathlib import Path
from typing import get_args

import amr_predict.evaluation as ae
import lightning as L
import numpy as np
import polars as pl
import torch
from amr_predict.models import MLP, Baseline
from amr_predict.preprocessing import EMBEDDING_METHODS
from amr_predict.utils import (
    TASK_TYPES,
    ModuleConfig,
    Preprocessor,
    data_spec,
    encode_strs,
    load_as,
    train_test_from_dict,
)
from datasets import Dataset
from loguru import logger
from xgboost import XGBClassifier, XGBRegressor

try:
    from snakemake.script import snakemake as smk
except ImportError:
    smk = type("snakemake", (), {"rule": None, "config": {}, "log": [1]})

logger.enable("amr_predict")
logger.add(smk.log[0])

RCONFIG = smk.config[smk.rule]
RNG: int = smk.config["rng"]
MODEL_ENV: dict = smk.config["models"]
os.environ["HF_HOME"] = smk.config["huggingface"]
torch.set_default_dtype(torch.float32)


RCONFIG["validation_kws"]["seed"] = RNG
DEFAULT_TRAIN = smk.config.get("trainer", {})
DEFAULT_LOADER = smk.config.get("dataloder", {})

if smk.rule == "cross_validate":
    RCONFIG["k_fold"]["random_state"] = RNG


def holdout_helper(
    dataset: Dataset,
    eva: ae.Evaluator,
    obs: pl.DataFrame,
    validation: Dataset | None = None,
) -> pl.DataFrame:
    holdout_splits = {}
    split_methods = {}
    for split_name, spec in RCONFIG["splits"].items():
        train_mask, test_mask = train_test_from_dict(df=obs, spec=spec)
        hsplit_lst = []
        for t, m in zip(["train", "test"], [train_mask, test_mask]):
            split_methods[f"{split_name}_{t}"] = m
            hsplit_lst.append(f"{split_name}_{t}")
        hsplit_lst.append(validation)
        holdout_splits[split_name] = hsplit_lst
    split_dset = ae.make_splits(dataset, split_methods=split_methods)
    return eva.holdout(split_dset, holdout_splits)


def randomize_dset(dset: Dataset, x_key: str) -> Dataset:
    "Generate dataset with randomized columns for testing"

    dset_dict = {x_key: dset[x_key][:]}

    for col in dset.column_names:
        was_tensor: bool = False
        if col != x_key:
            vals = dset[col][:]
            if torch.is_tensor(vals):
                was_tensor = True
                vals = vals.numpy()
            np.random.shuffle(vals)
            dset_dict[col] = vals if not was_tensor else torch.tensor(vals)
    dset = Dataset.from_dict(dset_dict).with_format("torch")
    return dset


def modify_for_test(dataset, x_key) -> Dataset:
    # dataset = concatenate_datasets([dataset, dataset, dataset])
    dataset = dataset.map(lambda x: {x_key: x[x_key][:1000]})
    return dataset


if smk.rule in {"cross_validate", "holdout"}:
    x_key, sample_key = (
        smk.config["pool_embeddings"]["key"],
        smk.config["pool_embeddings"]["sample_key"],
    )
    for dpath in smk.params["datasets"]:
        dname = Path(dpath).stem
        dataset: Dataset = load_as(dpath)
        if smk.config.get("test"):
            dataset = modify_for_test(dataset, x_key)
        obs: pl.DataFrame | None = (
            dataset.remove_columns("x").to_polars() if smk.rule == "holdout" else None
        )
        baseline_re = "|".join(map(lambda x: x + ".*", get_args(EMBEDDING_METHODS)))
        # The above re is fine because FM embeddings are named after pooling methods
        if re.match(baseline_re, dname):
            pp: Preprocessor | None = Preprocessor(
                x_key=x_key, **smk.config.get("baseline_filtering", {})
            )
        else:
            pp = None
        ttype: TASK_TYPES
        for ttype, task_names in smk.config["tasks"].items():
            if not task_names:
                continue
            if ttype == "classification":
                dataset, _ = encode_strs(dataset, task_names)
                in_features, n_classes = data_spec(dataset, y=task_names, x_key=x_key)
                bmodel = XGBClassifier
            else:
                in_features, n_classes = dataset[x_key][:].shape[1], None
                bmodel = XGBRegressor
            mconf = ModuleConfig(
                task_type=ttype,
                n_classes=n_classes,
                n_tasks=len(task_names),
                task_names=task_names,
            )
            for mname in RCONFIG["models"]:
                outfile = f"{smk.params["outdir"]}/{mname}/{dname}_{ttype}.csv"
                model_kws = (MODEL_ENV[mname] or {}).get("kws", {})
                trainer_kws = (MODEL_ENV[mname] or {}).get("trainer", DEFAULT_TRAIN)
                loader_kws = (MODEL_ENV[mname] or {}).get("dataloader", DEFAULT_LOADER)
                if mname == "baseline":
                    model = Baseline(
                        x_key=x_key,
                        device=smk.params["device"],
                        model=bmodel,
                        conf=mconf,
                        **model_kws,
                    )
                    valid_dset = None
                    validation_kws = None
                    trainer: L.Trainer | None = None
                elif mname == "mlp":
                    model = MLP(
                        in_features=in_features, x_key=x_key, conf=mconf, **model_kws
                    )
                    trainer = L.Trainer(**trainer_kws)
                eva_kws = dict(
                    model=model,
                    trainer=trainer,
                    preprocessor=pp,
                    **loader_kws,
                )
                if smk.rule == "cross_validate":
                    logger.info(f"Running cross validation with {mname}")
                    eva = ae.Evaluator(how="cv", **eva_kws)
                    result: pl.DataFrame = eva.cv(
                        dataset,
                        validation_kws=validation_kws,  # No need when using baseline
                        **RCONFIG["k_fold"],
                    )
                    logger.success("Cross validation complete")
                elif smk.rule == "holdout":
                    logger.info(f"Running holdout with {mname}")
                    eva = ae.Evaluator(how="holdout", **eva_kws)
                    result = holdout_helper(
                        dataset=dataset, eva=eva, obs=obs, validation=valid_dset
                    )
                    logger.success("Holdout complete")
                result.write_csv(outfile)
