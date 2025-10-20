#!/usr/bin/env ipython

from pathlib import Path

import amr_predict.evaluation as ae
import polars as pl
from amr_predict.models import Baseline
from amr_predict.utils import (
    TASK_TYPES,
    ModuleConfig,
    data_spec,
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
        smk.config["pooling"]["key"],
        smk.config["pooling"]["sample_key"],
    )
    for dpath in smk.params["datasets"]:
        dname = Path(dpath).stem
        dataset: Dataset = load_as(dpath)
        obs: pl.DataFrame | None = (
            dataset.remove_columns("x").to_polars() if smk.rule == "holdout" else None
        )
        for ttype, task_names in smk.config["tasks"].items():
            ttype: TASK_TYPES
            if not task_names:
                continue
            if ttype == "classification":
                in_features, n_classes = data_spec(dataset, y=task_names, x_key=x_key)
                bmodel = XGBClassifier
            else:
                in_features, n_classes = dataset[x_key][:].shape[1], None
                bmodel = XGBRegressor
            mconf = ModuleConfig(
                task_type=ttype, n_classes=n_classes, n_tasks=len(task_names)
            )
            for mname in RCONFIG["models"]:
                outfile = f"{smk.params["outdir"]}/{mname}/{dname}_{ttype}.csv"
                if mname == "baseline":
                    model = Baseline(
                        task_names=task_names,
                        x_key=x_key,
                        device=smk.params["device"],
                        model=bmodel,
                        conf=mconf,
                    )
                    valid_dset = None
                    validation_kws = None
                else:
                    raise NotImplementedError()
                if smk.rule == "cross_validate":
                    eva = ae.Evaluator(model=model, how="cv")
                    result: pl.DataFrame = eva.cv(
                        dataset,
                        validation_kws=validation_kws,  # No need when using baseline
                        **RCONFIG["k_fold"],
                    )
                elif smk.rule == "holdout":
                    eva = ae.Evaluator(model=model, how="holdout")
                    result = holdout_helper(
                        dataset=dataset, eva=eva, obs=obs, validation=valid_dset
                    )
                result.write_csv(outfile)
