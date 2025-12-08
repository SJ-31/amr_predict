#!/usr/bin/env ipython

import pytest
import tomllib
import yaml
from amr_predict.evaluation import Evaluator
from amr_predict.models import Baseline
from amr_predict.utils import ModuleConfig, data_spec, encode_strs, load_as
from pyhere import here
from xgboost import XGBClassifier, XGBRegressor

with open(here("tests", "env.toml"), "rb") as f:
    ENV: dict = tomllib.load(f)

with open(here("snakemake", "env.yaml"), "rb") as f:
    ENV.update(yaml.safe_load(f))

DIRS: dict = {
    "evo2": here("results", "tests", "with_evo2"),
    "seqlens": here("results", "tests", "no_date"),
}

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
