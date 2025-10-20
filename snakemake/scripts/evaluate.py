#!/usr/bin/env ipython

from pathlib import Path

import amr_predict.evaluation as ae
import polars as pl
from amr_predict.models import Baseline
from amr_predict.utils import TASK_TYPES, ModuleConfig, data_spec, load_as
from datasets import Dataset
from xgboost import XGBClassifier, XGBRegressor

try:
    from snakemake.script import snakemake as smk
except ImportError:
    smk = type("snakemake", (), {"rule": None, "config": {}})

RCONFIG = smk.config[smk.rule]
RNG: int = smk.config["rng"]

# def baseline_eval():

if smk.rule == "cross_validate":
    x_key, sample_key = (
        smk.config["pooling"]["key"],
        smk.config["pooling"]["sample_key"],
    )
    RCONFIG["validation_kws"]["seed"] = RNG
    RCONFIG["k_fold"]["random_state"] = RNG
    for dpath in smk.params["datasets"]:
        dname = Path(dpath).stem
        dataset: Dataset = load_as(dpath)
        for ttype, task_names in smk.config["tasks"].items():
            outfile = f"{smk.params["outdir"]}/{dname}_{ttype}.csv"
            if not task_names:
                continue
            ttype: TASK_TYPES
            if ttype == "classification":
                in_features, n_classes = data_spec(dataset, y=task_names, x_key=x_key)
                bmodel = XGBClassifier
            else:
                in_features, n_classes = dataset[x_key][:].shape[1], None
                bmodel = XGBRegressor
            mconf = ModuleConfig(
                task_type=ttype, n_classes=n_classes, n_tasks=len(task_names)
            )
            model = Baseline(
                task_names=task_names,
                x_key=x_key,
                device=smk.params["device"],
                model=bmodel,
                conf=mconf,
            )
            eva = ae.Evaluator(model=model, how="cv")
            result: pl.DataFrame = eva.cv(
                dataset,
                validation_kws=None,  # No need when using baseline
                **RCONFIG["k_fold"],
            )
            result.write_csv(outfile)
