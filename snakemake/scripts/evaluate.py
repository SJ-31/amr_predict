#!/usr/bin/env ipython

import os
from pathlib import Path

import amr_predict.evaluation as ae
import lightning as L
import polars as pl
from amr_predict.models import MLP, Baseline
from amr_predict.utils import (
    TASK_TYPES,
    ModuleConfig,
    data_spec,
    encode_strs,
    load_as,
    train_test_from_dict,
)
from datasets import Dataset
from xgboost import XGBClassifier, XGBRegressor

try:
    from snakemake.script import snakemake as smk
except ImportError:
    smk = type("snakemake", (), {"rule": None, "config": {}})

RCONFIG = smk.config[smk.rule]
RNG: int = smk.config["rng"]
MODEL_ENV: dict = smk.config["models"]
os.environ["HF_HOME"] = smk.config["huggingface"]

RCONFIG["validation_kws"]["seed"] = RNG

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


if smk.rule in {"cross_validate", "holdout"}:
    x_key, sample_key = (
        smk.config["pool_embeddings"]["key"],
        smk.config["pool_embeddings"]["sample_key"],
    )
    for dpath in smk.params["datasets"]:
        dname = Path(dpath).stem
        dataset: Dataset = load_as(dpath)
        obs: pl.DataFrame | None = (
            dataset.remove_columns("x").to_polars() if smk.rule == "holdout" else None
        )
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
                model_kws: dict = MODEL_ENV[mname].get("kws")
                model_kws = model_kws or {}
                trainer_kws = MODEL_ENV[mname].get("trainer")
                trainer_kws = trainer_kws or smk.config.get("trainer", {})
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
                if smk.rule == "cross_validate":
                    eva = ae.Evaluator(model=model, how="cv", trainer=trainer)
                    result: pl.DataFrame = eva.cv(
                        dataset,
                        validation_kws=validation_kws,  # No need when using baseline
                        **RCONFIG["k_fold"],
                    )
                elif smk.rule == "holdout":
                    eva = ae.Evaluator(model=model, how="holdout", trainer=trainer)
                    result = holdout_helper(
                        dataset=dataset, eva=eva, obs=obs, validation=valid_dset
                    )
                result.write_csv(outfile)
