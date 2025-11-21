#!/usr/bin/env ipython

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


def test_baseline_classification():
    dset = load_as(here(DIRS["seqlens"], "datasets", "pooled", "bin-mean"))
    dset, _ = encode_strs(dset, CLASSIFICATION_TASKS)
    in_features, n_classes = data_spec(dset, y=CLASSIFICATION_TASKS, x_key=X_KEY)
    mconf = ModuleConfig(
        task_type="classification",
        n_classes=n_classes,
        n_tasks=len(CLASSIFICATION_TASKS),
        task_names=CLASSIFICATION_TASKS,
    )
    model = Baseline(x_key=X_KEY, device="cpu", model=XGBClassifier, conf=mconf)
    eva: Evaluator = Evaluator(model=model)
    print(eva.holdout(dataset=dset))


def test_baseline_regression():
    dset = load_as(here(DIRS["seqlens"], "datasets", "pooled", "bin-mean"))
    in_features, n_classes = dset[X_KEY][:].shape[1], None
    mconf = ModuleConfig(
        task_type="regression",
        n_classes=n_classes,
        n_tasks=len(REGRESSION_TASKS),
        task_names=REGRESSION_TASKS,
    )
    model = Baseline(x_key=X_KEY, device="cpu", model=XGBRegressor, conf=mconf)
    eva = Evaluator(model=model)
    print(eva.holdout(dset))
