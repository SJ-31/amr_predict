#!/usr/bin/env ipython

import copy
from pathlib import Path

import numpy as np
import pytest
import tomllib
import yaml
from amr_predict.evaluation import Evaluator
from amr_predict.models import Baseline
from amr_predict.utils import (
    ModuleConfig,
    Preprocessor,
    data_spec,
    encode_strs,
    load_as,
)
from loguru import logger
from pyhere import here
from xgboost import XGBClassifier, XGBRegressor

with open(here("tests", "env.toml"), "rb") as f:
    ENV: dict = tomllib.load(f)

with open(here("snakemake", "env.yaml"), "rb") as f:
    ENV.update(yaml.safe_load(f))

REMOTE = here("data", "remote")
JIA = here(REMOTE, "2025-10-22_jia_seqlens", "datasets")

DIRS: dict = {
    "evo2": here("results", "tests", "with_evo2"),
    "seqlens": here("results", "tests", "no_date"),
}

logger.enable("amr_predict")

X_KEY, SAMPLE_KEY = (
    ENV["pool_embeddings"]["key"],
    ENV["pool_embeddings"]["sample_key"],
)

REGRESSION_TASKS = ["AMK", "GEN"]
CLASSIFICATION_TASKS = ["AMK", "GEN"]


@pytest.mark.skip(reason="passed")
@pytest.mark.parametrize(
    "task_type,tasks",
    [("classification", CLASSIFICATION_TASKS), ("regression", REGRESSION_TASKS)],
)
def test_baseline(task_type, tasks):
    dset = load_as(here(DIRS["seqlens"], "datasets", "pooled", "bin-mean"))
    dset, _ = encode_strs(dset, tasks)
    if task_type == "classification":
        model = XGBClassifier
    else:
        model = XGBRegressor
    in_features, n_classes = data_spec(dset, y=tasks, x_key=X_KEY)
    mconf = ModuleConfig(
        task_type=task_type,
        n_classes=n_classes,
        n_tasks=len(tasks),
        task_names=tasks,
    )
    model = Baseline(x_key=X_KEY, device="cpu", model=model, conf=mconf)
    eva: Evaluator = Evaluator(model=model)
    print(eva.holdout(dataset=dset))


def test_pp(tmp_path):
    dataset = load_as(here(JIA, "pooled", "bin-mean"), "huggingface")
    ddict = dataset.train_test_split()
    feature_file: Path = tmp_path / "features.txt"
    pp = Preprocessor(
        method="variance", x_key="x", feature_file=feature_file, quantile_threshold=0.5
    )
    logger.info(f"original shape: {dataset["x"][:].shape}")
    t1 = pp.fit_transform(ddict["train"])
    logger.info(f"after filtering: {t1["x"][:].shape}")
    saved = copy.deepcopy(pp.feature_idx)
    indices = feature_file.read_text().splitlines()
    logger.info(f"saved indices: {indices[1:10]}...")
    t2 = pp.transform(ddict["test"])
    assert np.all(saved == pp.feature_idx)
    assert t1["x"][:].shape[1] == t2["x"][:].shape[1]
