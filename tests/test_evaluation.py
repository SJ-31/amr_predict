#!/usr/bin/env ipython

import copy
from pathlib import Path

import numpy as np
import pytest
import tomllib
import yaml
from amr_predict.evaluation import Evaluator, make_control_task
from amr_predict.models import Baseline
from amr_predict.utils import (
    ModuleConfig,
    Preprocessor,
    data_spec,
    encode_strs,
    load_as,
    read_tabular,
    with_metadata,
)
from datasets import Dataset
from loguru import logger
from pyhere import here
from xgboost import XGBClassifier, XGBRegressor

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


@pytest.fixture
def dset() -> Dataset:
    loaded: Dataset = load_as(
        here(DIRS["seqlens"], "datasets", "pooled", "bin-mean"), "huggingface"
    )
    return loaded


def test_add_ctrl(toy_dset, env):
    obs = read_tabular(env["sample_metadata"]["file"])
    dset = toy_dset(
        obs["BioSample"][:500],
        500,
        {"amr_class": ["resistant", "susceptible", "intermediate"]},
    )
    assert dset.shape[0] == 500
    dset = with_metadata(
        dset,
        env,
        sample_col="sample",
        meta_options=("sample",),
    )
    assert dset.shape[0] == 500
    assert None not in dset["amr_class"][:]
    mapping: dict = make_control_task(
        dset, target_task="amr_class", control_col="species"
    )
    assert set(mapping.keys()) == set(dset["species"])
    assert set(mapping.values()) == set(dset["amr_class"])
    old_target = list(dset["amr_class"][:])
    dset = make_control_task(
        dset, target_task="amr_class", control_col="species", add=True, added_name="foo"
    )
    assert list(dset["foo"][:]) != old_target


@pytest.mark.skip(reason="passed")
@pytest.mark.parametrize(
    "task_type,tasks",
    [("classification", CLASSIFICATION_TASKS), ("regression", REGRESSION_TASKS)],
)
def test_baseline(dset, task_type, tasks):
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
    model = Baseline(x_key=X_KEY, device="cpu", model=model, cfg=mconf)
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
