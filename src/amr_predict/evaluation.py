#!/usr/bin/env ipython

import numpy as np
from torch.utils.data import DataLoader
from typing import Literal

import polars as pl
import sklearn.model_selection as ms
import torchmetrics as tmet
from datasets import Dataset
from sklearn.base import BaseEstimator


class Evaluator:
    def __init__(self, model, how: Literal["cv", "holdout"] = "cv") -> None:
        self.with_sklearn: bool = isinstance(model, BaseEstimator)
        self.model = model
        self.how = how

    def _ds_setup(self, dataset: Dataset) -> DataLoader | tuple[np.ndarray, np.ndarray]:
        

    def cv(dataset: Dataset) -> pl.DataFrame: ...

    def holdout(dataset: Dataset) -> pl.DataFrame: ...
