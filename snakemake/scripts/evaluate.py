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
    with_metadata,
)
from datasets import Dataset, DatasetDict
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

X_KEY, SAMPLE_KEY = (
    smk.config["pool_embeddings"]["key"],
    smk.config["pool_embeddings"]["sample_key"],
)

RCONFIG["validation_kws"]["seed"] = RNG
DEFAULT_TRAIN = smk.config.get("trainer", {})
DEFAULT_LOADER = smk.config.get("dataloader", {})

if smk.rule == "cross_validate":
    RCONFIG["k_fold"]["random_state"] = RNG


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


def make_eval_kws(
    model_name, bmodel, module_cfg, preprocessor, in_features
) -> tuple[dict, dict]:
    model_kws = (MODEL_ENV[model_name] or {}).get("kws", {})
    trainer_kws = (MODEL_ENV[model_name] or {}).get("trainer", DEFAULT_TRAIN)
    loader_kws = (MODEL_ENV[model_name] or {}).get("dataloader", DEFAULT_LOADER)
    val_kws = RCONFIG.get("validation_kws", {})
    if model_name == "baseline":
        model = Baseline(
            x_key=X_KEY,
            device=smk.params["device"],
            model=bmodel,
            cfg=module_cfg,
            **model_kws,
        )
        trainer: L.Trainer | None = None
        val_kws = {}
    elif model_name == "mlp":
        model = MLP(in_features=in_features, x_key=X_KEY, cfg=module_cfg, **model_kws)
        trainer = L.Trainer(**trainer_kws)
    else:
        raise ValueError(f"model name {model_name} not recognized")
    return dict(
        model=model,
        trainer=trainer,
        preprocessor=preprocessor,
        **loader_kws,
    ), val_kws


# * Rule functions
def holdout(
    evaluator_kws: dict, dataset: Dataset, model_name: str, validation_kws: dict
) -> pl.DataFrame:
    logger.info(f"Running holdout with {model_name}")
    holdout_splits = {}
    split_methods = {}
    eva = ae.Evaluator(how="holdout", **evaluator_kws)
    obs: pl.DataFrame | None = dataset.remove_columns(X_KEY).to_polars()
    for split_name, spec in RCONFIG["splits"].items():
        train_mask, test_mask = train_test_from_dict(df=obs, spec=spec)
        hsplit_lst = []
        for t, m in zip(["train", "test"], [train_mask, test_mask]):
            split_methods[f"{split_name}_{t}"] = m
            hsplit_lst.append(f"{split_name}_{t}")
        holdout_splits[split_name] = hsplit_lst
    split_dset: DatasetDict = ae.make_splits(dataset, split_methods=split_methods)
    result = eva.holdout(split_dset, holdout_splits, validation_kws=validation_kws)
    logger.info(f"Holdout with {model_name} complete")
    return result


def cv_wrapper(
    evaluator_kws,
    dataset: Dataset,
    validation_kws,
    add_control_tasks: bool = False,
) -> pl.DataFrame:
    if add_control_tasks and (ctask_spec := RCONFIG["control_tasks"]):
        for target, control in ctask_spec.items():
            dataset = ae.make_control_task(
                dataset,
                target_task=target,
                control_col=control,
                seed=RNG,
                add=True,
                added_name=target,  # Just replace the target column with the randomized control
            )
    eva = ae.Evaluator(how="cv", **evaluator_kws)
    result: pl.DataFrame = eva.cv(
        dataset,
        validation_kws=validation_kws,  # No need when using baseline
        **RCONFIG["k_fold"],
    )
    return result


def cross_validate(model_name, **kws) -> pl.DataFrame:
    logger.info(f"Running cross validation with {model_name}")
    result = cv_wrapper(**kws, add_control_tasks=False)
    logger.success("Cross validation complete")
    return result


def cv_control_tasks(model_name, **kws) -> pl.DataFrame:
    logger.info(f"Running cv control tasks with {model_name}")
    result = cv_wrapper(**kws, add_control_tasks=True)
    logger.success("Cross validation with control tasks complete")
    return result


# * Entry


if smk.rule in {"cross_validate", "holdout"}:
    for dpath in smk.params["datasets"]:
        dname = Path(dpath).stem
        dataset: Dataset = load_as(dpath)
        dataset = with_metadata(dataset, smk.config, meta_options=("ast", "sample"))
        if smk.config.get("test"):
            dataset = modify_for_test(dataset, X_KEY)
        baseline_re = "|".join(map(lambda x: x + ".*", get_args(EMBEDDING_METHODS)))
        # The above re is fine because FM embeddings are named after pooling methods
        if re.match(baseline_re, dname):
            pp: Preprocessor | None = Preprocessor(
                x_key=X_KEY, **smk.config.get("baseline_filtering", {})
            )
        else:
            pp = None
        ttype: TASK_TYPES
        for ttype, task_names in smk.config["tasks"].items():
            if not task_names:
                continue
            if ttype == "classification":
                dataset, _ = encode_strs(dataset, task_names)
                in_features, n_classes = data_spec(dataset, y=task_names, x_key=X_KEY)
                bmodel = XGBClassifier
            else:
                in_features, n_classes = dataset[X_KEY][:].shape[1], None
                bmodel = XGBRegressor
            mconf = ModuleConfig(
                task_type=ttype,
                n_classes=n_classes,
                n_tasks=len(task_names),
                task_names=task_names,
            )
            for mname in RCONFIG["models"]:
                outfile = f"{smk.params["outdir"]}/{mname}/{dname}_{ttype}.csv"
                if Path(outfile).exists():
                    continue
                eva_kws, validation_kws = make_eval_kws(
                    mname,
                    bmodel=bmodel,
                    module_cfg=mconf,
                    preprocessor=pp,
                    in_features=in_features,
                )
                fn = globals()[smk.rule]
                result = fn(
                    evaluator_kws=eva_kws,
                    model_name=mname,
                    dataset=dataset,
                    validation_kws=validation_kws,
                )
                result.write_csv(outfile)
