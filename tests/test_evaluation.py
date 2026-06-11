#!/usr/bin/env ipython

import copy
from pathlib import Path

import lightning as L
import numpy as np
import polars as pl
import pytest
import torch
from amr_predict.evaluation import (
    EvalSAE,
    Evaluator,
    LatentAblation,
    make_control_task,
    max_by_label,
)
from amr_predict.metadata import encode_strs, with_metadata
from amr_predict.metrics import multitask_all_cls
from amr_predict.models import MLP, Baseline
from amr_predict.sae import BatchTopK
from amr_predict.utils import (
    ModuleConfig,
    Preprocessor,
    data_spec,
    load_as,
    read_tabular,
)
from datasets import Dataset, concatenate_datasets, disable_progress_bar
from loguru import logger
from pyhere import here
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from torch.utils.data import DataLoader
from xgboost import XGBClassifier, XGBRegressor

disable_progress_bar()

DIRS: dict = {
    "evo2": here("results", "tests", "with_evo2"),
    "seqlens": here("results", "tests", "no_date"),
}


logger.enable("amr_predict")


REGRESSION_TASKS = ["AMK", "GEN"]
CLASSIFICATION_TASKS = ["AMK", "GEN"]

torch.set_default_dtype(torch.float64)


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


@pytest.mark.parametrize("how", ["embedding", "reconstruction", "activation"])
def test_sae_ablate(random_linked_dset, how):
    dummy_embeddings = random_linked_dset(dim=100)

    cfg: ModuleConfig = ModuleConfig(act_size=100, dict_size=1000, top_k=5)
    trainer = L.Trainer(max_epochs=20, enable_progress_bar=False)
    sae = BatchTopK(cfg, x_key="x")
    loader = DataLoader(dummy_embeddings, batch_size=10)
    trainer.fit(sae, train_dataloaders=loader)

    ablator = LatentAblation(eva=Evaluator(RandomForestClassifier()), x_key="x")


def test_mlp_cls(toy_dset):
    dset = toy_dset(
        {"amr_class": ["resistant", "susceptible", "intermediate"]},
        seq_level=True,
        n=1000,
        x_size=200,
    )
    task_names = ["amr_class"]
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
    trainer = L.Trainer(max_epochs=20)

    # Simple test of overfitting to train
    eva = Evaluator(
        model=lambda x: MLP(
            in_features=x[1], cfg=mconf, num_layers=2, x_key="x", hidden_dim=50
        ),
        task_names=None,
        trainer=trainer,
        kws={"batch_size": 5},
    )
    print(eva.cv(dset, n_splits=3, n_repeats=2))


def test_permute(rng):
    random = torch.hstack(
        [
            torch.randint(3, (10, 1)),
            torch.randint(8, (10, 1)),
            torch.randint(200, (10, 1)),
        ]
    )
    y = rng.choice(list("abc"), size=10)
    logger.info("original {}", random)
    to_fill = torch.zeros_like(random)
    for _, group in pl.DataFrame({"y": y}).with_row_index().group_by("y"):
        indices = group["index"]
        cur = torch.tensor(rng.permuted(random[indices, :], axis=0))
        to_fill[indices, :] = cur
    logger.info("Permuted {}", to_fill)


@pytest.mark.parametrize("ttype", ["class_label", "per_class_feature"])
def test_permutation_test(ttype, toy_dset):
    dset = toy_dset(n=500, column_spec={"y": list("abcdef")}, seq_level=True, x_size=5)
    _, n_classes = data_spec(dset, y=["y"], x_key="x")
    dset, _ = encode_strs(dset, ["y"])
    eva = Evaluator(
        model=RandomForestClassifier(),
        task_names=["y"],
        x_key="x",
        task_type="classification",
        n_classes=n_classes,
        verbose=False,
    )
    test = eva.permutation_test(
        dset, test_type=ttype, use_cv=True, n_splits=3, n_repeats=2
    )
    logger.info("test {}", test)


@pytest.mark.parametrize(
    "task_type,cspec",
    [
        ("classification", {"y": [1, 0]}),
        ("regression", {"y": lambda rng, s: rng.random(size=s)}),
    ],
)
def test_evaluator_sklearn(task_type, cspec, toy_dset):
    dset = toy_dset(n=2000, column_spec=cspec)
    in_features, n_classes = data_spec(dset, y=["y"], x_key="x")
    model = (
        RandomForestRegressor
        if task_type != "classification"
        else RandomForestClassifier
    )
    eva = Evaluator(
        model=model,
        x_key="x",
        task_names=["y"],
        task_type=task_type,
        n_classes=n_classes,
    )
    holdout = eva.holdout(dataset=dset)
    cv = eva.cv(dataset=dset, n_splits=5, n_repeats=3)
    logger.info("holdout: {}", holdout)
    logger.info("cv: {}", cv)


@pytest.mark.parametrize(
    "task_type,tasks,cspec",
    [
        (
            "classification",
            CLASSIFICATION_TASKS,
            {"AMK": ["R", "S", "M"], "GEN": ["R", "S", "M"]},
        ),
        (
            "regression",
            REGRESSION_TASKS,
            {
                "AMK": lambda rng, s: rng.random(size=s),
                "GEN": lambda rng, s: rng.random(size=s) + 500,
            },
        ),
    ],
)
def test_baseline(toy_dset, task_type, tasks, cspec, keys):
    dset = toy_dset(n=2000, column_spec=cspec)
    x_key, sample_key = keys
    if task_type == "classification":
        model = XGBClassifier
        dset, _ = encode_strs(dset, tasks)
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
    eva: Evaluator = Evaluator(
        model=model, task_names=None, n_classes=None, task_type=None
    )
    holdout = eva.holdout(dataset=dset)
    cv = eva.cv(dataset=dset, n_splits=5, n_repeats=2)
    logger.info("holdout: {}", holdout)
    logger.info("cv: {}", cv)


# def test_latent_ablation() ->


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


def test_compute_metrics():
    acts = ""


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
    labels = pl.DataFrame(
        {"labels": ["A", "A", "A", "B", "B", "B"], "sample": range(6)}
    )
    eval = EvalSAE(acts, threshold=0.3)
    scores = eval.score_latents(labels, label_col="labels")
    report = scores.report(k=1)
    assert report["label"].to_list() == ["B", "A", "A", "B", "B"]
    assert report["sensitivity"].to_list() == pytest.approx(
        [2 / 3, 2 / 3, 2 / 3, 1 / 3, 2 / 3]
    )


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
        model=model_fn,
        preprocessor=pp,
        trainer=L.Trainer(max_epochs=5),
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
