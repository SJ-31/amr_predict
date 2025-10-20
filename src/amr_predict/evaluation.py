#!/usr/bin/env ipython

from pathlib import Path
from typing import Callable, Literal, TypeAlias

import lightning as L
import numpy as np
import polars as pl
import sklearn.model_selection as ms
import torch.nn as nn
from amr_predict.metrics import (
    multitask_all_cls,
    multitask_all_reg,
    multitask_metrics2df,
)
from amr_predict.models import Baseline
from amr_predict.utils import TASK_TYPES, load_as
from datasets import Dataset, DatasetDict
from torch import Tensor
from torch.utils.data import DataLoader

MODEL_CLASSES: TypeAlias = L.LightningModule | Baseline | nn.Module


class Evaluator:
    def __init__(
        self,
        model: MODEL_CLASSES,
        how: Literal["cv", "holdout"] = "cv",
        trainer: L.Trainer | None = None,
        **kws,
    ) -> None:
        self.model: MODEL_CLASSES = model
        self.x_key: str = self.model.x_key
        self.task_type: TASK_TYPES = self.model.conf.task_type
        self.trainer: L.Trainer | None = trainer
        self.how: str = how
        self.kws: dict = kws

    def _fit(self, train: Dataset, val: Dataset | None = None) -> None:
        if isinstance(self.model, Baseline):
            self.model.fit(train)
        elif self.trainer is None:
            raise ValueError("Trainer must be provided if not using baseline model")
        else:
            tl = DataLoader(train, **self.kws)
            vl = DataLoader(val, **self.kws) if val is not None else val
            self.trainer.fit(self.model, train_dataloaders=tl, val_dataloaders=vl)

    def cv(
        self,
        dataset: Dataset,
        validation_kws: dict | None = None,
        stratify_by: str | None = None,
        **kws,
    ) -> pl.DataFrame:
        """
        K-fold cross-validation
        [2025-10-16 Thu] WARNING: datasets.select might create new datasets, but you checked locally
        and it seemed not to

        Parameters
        ----------
        validation : float
            optional, the proportion of the dataset to keep for validation
        stratify_by : str | None
            for classification tasks, the name of the task (column in dataset) to stratify
        kws : dict
            kws passed to the sklearn k-fold init
        """
        validation_kws = validation_kws or {}
        if not validation_kws:
            val_split = dataset.train_test_split(**validation_kws)
            val: Dataset | None = val_split["test"]
            dataset = val_split["train"]
        else:
            val = None
        if self.task_type == "classification":
            y = dataset[stratify_by][:] if stratify_by else None
            k_fold = ms.RepeatedStratifiedKFold(**kws)
            splits = k_fold.split(np.zeros_like(dataset[self.x_key][:]), y=y)
        else:
            k_fold = ms.RepeatedKFold(**kws)
            splits = k_fold.split(np.zeros_like(dataset[self.x_key][:]))
        split_indices, split_names = {}, {}
        for i, (train, test) in enumerate(splits):
            fname = f"fold_{i}"
            train_key, test_key = f"{fname}_train", f"{fname}_test"
            if validation_kws:
                split_names[fname] = (train_key, test_key, val)
            else:
                split_names[fname] = (train_key, test_key, None)
            split_indices[train_key] = train
            split_indices[test_key] = test

        ddict = make_splits(dataset=dataset, split_methods=split_indices)
        return self.holdout(ddict, split_names)

    def holdout(
        self,
        dataset: Path | DatasetDict,
        splits: dict[str, tuple[str, str, str | None | Dataset]],
    ) -> pl.DataFrame:
        """Holdout evaluation on a dataset dict, possibly saved on disk

        Parameters
        ----------
        splits : Sequence of (train, test, validation) tuple names. Validation set is
            optional

        Returns
        -------
        DataFrame of evaluation results
        """
        results: list[pl.DataFrame] = []
        tasks = self.model.task_names
        for key, (train, test, val) in splits.items():
            if isinstance(val, str) and isinstance(dataset, Path):
                val_dset: Dataset | None = load_as(dataset / val)
            elif isinstance(val, Dataset):
                val_dset = val
            else:
                val_dset = dataset.get(val)
            if isinstance(dataset, Path):
                train_dset: Dataset = load_as(dataset / train)
                test_dset: Dataset = load_as(dataset / test)
            else:
                train_dset = dataset[test]
                test_dset = dataset[test]
            self._fit(train=train_dset, val=val_dset)
            y_true: Tensor = test_dset.to_polars().select(tasks).to_torch()
            if self.task_type == "regression":
                y_pred: Tensor = self.model.predict_step(test_dset)
                metrics = multitask_all_reg(y_pred, y_true, task_names=tasks)
            else:
                y_pred: Tensor = self.model.predict_proba(test_dset)
                metrics = multitask_all_cls(
                    y_pred,
                    y_true,
                    n_classes=self.model.conf.n_classes,
                    task_names=tasks,
                )
            df = multitask_metrics2df(metrics)
            results.append(df.with_columns(pl.lit(key).alias("test_set")))
        return pl.concat(results)


def make_splits(
    dataset: Dataset, split_methods: dict[str, Callable | np.ndarray | float]
) -> DatasetDict:
    """Helper function for splitting a hf dataset into a dataset dict

    Parameters
    ----------
    split_methods : dict[str, Callable | np.ndarray]
        dict where keys are the names of the splits, and values are either
        1. A function compatible with dataset.filter
        2. An array of indices, or a boolean mask
        3. A float, in which case dataset.train_test_split is called and the train, test
            splits are saved by treating the key as a suffix
    """
    indices = np.array(range(len(dataset)))
    result: DatasetDict = DatasetDict()
    for key, spec in split_methods.items():
        if isinstance(spec, Callable):
            result[key] = dataset.filter(spec)
        elif isinstance(spec, float):
            split = dataset.train_test_split(spec)
            result[f"{key}_test"] = split["test"]
            result[f"{key}_train"] = split["train"]
        elif isinstance(spec, np.ndarray) and spec.dtype == bool:
            ranges = indices[spec]
            result[key] = dataset.select(ranges)
        else:
            result[key] = dataset.select(spec)
    return result
