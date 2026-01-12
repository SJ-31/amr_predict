#!/usr/bin/env ipython

import copy
from pathlib import Path

import lightning as L
import numpy as np
import pytest
import torch
from amr_predict.evaluation import EvalSAE, Evaluator, make_control_task, max_by_label
from amr_predict.metrics import multitask_all_cls
from amr_predict.models import MLP, Baseline
from amr_predict.utils import (
    ModuleConfig,
    Preprocessor,
    data_spec,
    encode_strs,
    load_as,
    read_tabular,
    with_metadata,
)
from datasets import Dataset, concatenate_datasets
from loguru import logger
from pyhere import here
from torch.utils.data import DataLoader
from xgboost import XGBClassifier, XGBRegressor

DIRS: dict = {
    "evo2": here("results", "tests", "with_evo2"),
    "seqlens": here("results", "tests", "no_date"),
}


logger.enable("amr_predict")


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
        {"amr_class": ["resistant", "susceptible", "intermediate"]},
        obs["BioSample"][:500],
        n=500,
    )
    assert dset.shape[0] == 500
    dset = with_metadata(
        dset,
        env,
        sample_col="sample",
        meta_options=("sample",),
    )
    dset, _ = encode_strs(dset, ("amr_class",))
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


def test_mlp_cls():
    dset = load_as(
        here("results", "tests", "esm_test", "datasets", "pooled", "orf_only-esm-mean")
    ).with_format("torch")
    task_names = ["amikacin_class", "gentamicin_class"]
    for t in task_names:
        dset = dset.add_column(
            t, np.random.choice(["R", "I", "S"], dset.shape[0], replace=True)
        )
    in_features, n_classes = data_spec(dset, y=task_names, x_key="x")
    mconf = ModuleConfig(
        task_type="classification",
        n_tasks=len(task_names),
        task_names=task_names,
        n_classes=n_classes,
        record_metrics=True,
    )
    dset, _ = encode_strs(dset, task_names)
    dset = dset.select_columns(["x"] + task_names)
    model = MLP(
        in_features=in_features, cfg=mconf, num_layers=2, x_key="x", hidden_dim=50
    )
    trainer = L.Trainer(max_epochs=20)
    loader = DataLoader(dset, batch_size=5)

    # Simple test of overfitting to train
    trainer.fit(model, train_dataloaders=loader)
    y_pred = model.predict_proba(dset)
    y_true = dset.to_polars().select(task_names).to_torch()
    metrics = multitask_all_cls(
        y_pred,
        y_true,
        n_classes=model.cfg.n_classes,
        task_names=task_names,
    )
    print(metrics)


@pytest.mark.skip(reason="passed")
@pytest.mark.parametrize(
    "task_type,tasks",
    [("classification", CLASSIFICATION_TASKS), ("regression", REGRESSION_TASKS)],
)
def test_baseline(dset, task_type, tasks, keys):
    x_key, sample_key = keys
    dset, _ = encode_strs(dset, tasks)
    if task_type == "classification":
        model = XGBClassifier
    else:
        model = XGBRegressor
    in_features, n_classes = data_spec(dset, y=tasks, x_key=x_key)
    mconf = ModuleConfig(
        task_type=task_type,
        n_classes=n_classes,
        n_tasks=len(tasks),
        task_names=tasks,
    )
    model = Baseline(x_key=x_key, device="cpu", model=model, cfg=mconf)
    eva: Evaluator = Evaluator(model=model)
    print(eva.holdout(dataset=dset))


def test_max_by_lab():
    acts = torch.tensor(
        [
            [0.1, 0.9, 0.2, 0.3, 0.1],
            [0.2, 0.3, 0.9, 0.0, 0.3],
            [0.9, 0.2, 0.3, 0.1, 0.1],
            [0.2, 0.1, 0.3, 0.8, 0.2],
            [0.8, 0.2, 0.1, 0.2, 0.9],
            [0.3, 0.3, 0.7, 0.1, 0.3],
        ]
    )
    labels = ["A", "A", "A", "B", "B", "B"]
    ans = max_by_label(acts, labels)
    assert ans.shape[0] == acts.shape[1]
    assert (
        ans
        == torch.tensor([[0.9, 0.8], [0.9, 0.3], [0.9, 0.7], [0.3, 0.8], [0.3, 0.9]])
    ).all()


def test_score_latents():
    acts = torch.tensor(
        [
            [0.1, 0.9, 0.2, 0.3, 0.1],
            [0.2, 0.3, 0.9, 0.0, 0.3],
            [0.9, 0.2, 0.3, 0.1, 0.1],
            [0.2, 0.1, 0.3, 0.8, 0.2],
            [0.8, 0.2, 0.1, 0.2, 0.9],
            [0.3, 0.3, 0.7, 0.1, 0.3],
        ]
    )
    labels = ["A", "A", "A", "B", "B", "B"]
    eval = EvalSAE(acts)
    scores = eval.score_latents(labels, active_threshold=0.2)
    assert scores["label_max"].to_list() == ["B", "A", "A", "B", "B"]
    assert scores["sensitivity"].to_list() == [2 / 3, 2 / 3, 2 / 3, 1 / 3, 2 / 3]


@pytest.mark.parametrize(
    "fail,with_model_fn",
    [
        (True, False),
        (False, True),
    ],
)
def test_mlp_pp(keys, env, with_model_fn, fail):
    x_key, sample_key = keys
    dataset = load_as(
        here("results", "tests", "esm_test", "datasets", "pooled", "kmer10")
    )
    dataset = with_metadata(dataset, env, "sample", ("ast",)).with_format("torch")
    dataset = concatenate_datasets([dataset, dataset, dataset])
    pp = Preprocessor(method="variance", x_key="x", quantile_threshold=0.5)
    # print(dataset)
    in_features, n_classes = dataset["x"][:].shape[1], None
    cfg = ModuleConfig(
        task_type="regression",
        task_names=("amikacin", "gentamicin"),
    )
    model = MLP(in_features=in_features, x_key=x_key, cfg=cfg)
    model_fn = None
    if with_model_fn:
        model = MLP(in_features=in_features, x_key=x_key, cfg=cfg, batch_norm=True)
        model_fn = lambda x: MLP(
            in_features=x, x_key=x_key, cfg=cfg, num_layers=2, batch_norm=True
        )
    eva = Evaluator(
        model=model,
        preprocessor=pp,
        trainer=L.Trainer(max_epochs=5),
        model_fn=model_fn,
        batch_size=5,
        drop_last=True,
    )
    if fail:
        with pytest.raises(
            RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"
        ):
            eva.cv(dataset)
    else:
        eva.cv(dataset)


def test_pp(tmp_path, rdset):
    torch.set_default_dtype(torch.float64)
    dataset = load_as(here(rdset("mora"), "pooled", "bin-mean"), "huggingface")
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
