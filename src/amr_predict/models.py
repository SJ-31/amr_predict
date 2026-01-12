#!/usr/bin/env ipython

from collections.abc import Callable, Iterable, Sequence
from typing import override

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schedule
import torchmetrics as tmet
from amr_predict.metrics import multitask_all_reg, multitask_cross_entropy_loss
from amr_predict.utils import CACHE_OPTIONS, TASK_TYPES, ModuleConfig, iter_cols
from datasets.arrow_dataset import Dataset
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from torchmetrics.classification import Accuracy
from xgboost import XGBClassifier

# * Baseline


class Baseline:
    """A baseline class for multitask prediction. Consists of XGBoost models
    trained independently on each task
    """

    def __init__(
        self,
        x_key: str = "x",
        device: str | torch.device = "cpu",
        model=XGBClassifier,
        cfg: ModuleConfig | None = None,
        **kws,
    ):
        self.x_key = x_key
        self.task_names: Sequence | str = cfg.task_names
        self.models: list = [model(**kws) for _ in range(cfg.n_tasks)]
        self.cfg: ModuleConfig = ModuleConfig() if cfg is None else cfg
        self.task_type: TASK_TYPES = self.cfg.task_type
        self.device: torch.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

    def fit(self, x: Dataset | DataLoader):
        X: Dataset = x if isinstance(x, Dataset) else x.dataset
        x_tensor = X[self.x_key][:]
        y = X.to_polars().select(self.task_names).to_numpy()
        X = x_tensor.cpu().numpy()
        for model, y in zip(self.models, iter_cols(y)):
            model.fit(X, y)

    def _predict_helper(self, batch, proba: bool = False) -> Tensor | tuple:
        try:
            x, _ = batch
        except ValueError:
            x = batch
        if isinstance(x, DataLoader):
            x = x.dataset[self.x_key][:]
        elif isinstance(x, Dataset):
            x = x[self.x_key][:]
        elif isinstance(x, dict):
            x = x[self.x_key]
        x = x.cpu().numpy()
        if proba:
            predictions = tuple(m.predict_proba(x) for m in self.models)
        else:
            predictions = tuple(m.predict(x) for m in self.models)
        if self.task_type != "classification":
            predictions = torch.tensor(np.column_stack(predictions), device=self.device)
        else:
            predictions = tuple(
                torch.tensor(p, device=self.device) for p in predictions
            )
        return predictions

    def predict_step(self, batch):
        return self._predict_helper(batch, False)

    def predict_proba(self, batch):
        return self._predict_helper(batch, True)


# * BaseNN


class BaseNN(L.LightningModule):
    def __init__(
        self,
        in_features: int,
        x_key: str = "x",
        cfg: ModuleConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        in_features : int
            number of incoming features
        """
        super().__init__()
        self.in_features: int = in_features
        self.x_key: str = x_key
        self.cfg: ModuleConfig = ModuleConfig() if cfg is None else cfg
        self.n_tasks: int = self.cfg.n_tasks
        self.metric_loggers: nn.ModuleList
        self.supervised: bool = True
        self.task_type: TASK_TYPES = self.cfg.task_type
        self.n_classes: tuple[int] = (
            (1,) if self.task_type == "regression" else self.cfg.n_classes
        )
        if self.cfg.record and self.task_type == "classification":
            self.metric_loggers = nn.ModuleList(
                [Accuracy(task="multiclass", num_classes=n) for n in self.n_classes]
            )
        elif self.cfg.record and self.task_type == "regression":
            self.metric_loggers = nn.ModuleList(
                [MeanSquaredError(num_outputs=self.n_tasks)]
            )

        if self.cfg.task_names is None:
            self.task_names: Sequence[str] = [str(i) for i in range(self.n_tasks)]
        else:
            self.task_names = self.cfg.task_names

        # Cache results after iterations or validation for custom callbacks
        self.cache: dict[str, tuple[bool, list]] = {
            "train_loss": (False, []),
            "train_acc": (False, []),
            "val_acc": (False, []),
            "val_loss": (False, []),
            "test_loss": (False, []),
            "test_acc": (False, []),
        }
        if isinstance(self.cfg.cache, str):
            self.set_cache(self.cfg.cache)
        elif self.cfg.cache is not None:
            for c in self.cfg.cache:
                self.set_cache(c)

    def get_outlayer(self, i: int) -> nn.Module:
        """Return the module's ith outlayer"""
        raise NotImplementedError()

    @staticmethod
    def iter_cols(x: Tensor | np.ndarray | tuple) -> Iterable:
        """Iterate over columnes of x"""
        if isinstance(x, Tensor):
            to_iter = torch.unbind(x, dim=1)
        elif isinstance(x, np.ndarray):
            to_iter = [x[:, i] for i in range(x.shape[1])]
        else:
            to_iter = iter(x)
        return to_iter

    @classmethod
    def new(cls, in_features: int, x_key: str, conf: ModuleConfig | None, **kws):
        return cls(in_features, x_key, conf, **kws)

    def set_cache(self, value: CACHE_OPTIONS):
        if value not in self.cache:
            raise ValueError(f"Value to cache must be one of {self.cache.keys()}")
        self.cache[value] = (True, [])

    def _try_cache_to(self, target: str, value: Tensor) -> None:
        """Record ``value`` to the cache if it has been set for recording"""
        if self.cache[target][0]:
            self.cache[target][1].append(value.detach())

    def cache_clear(self, target) -> None:
        self.cache[target][1].clear()

    def register_optimizers(self, opt_fn: Callable):
        """Specify the optimizer to use for this model

        Parameters
        ----------
        opt_fn : Callable
            returns a Pytorch-compatible optimizer when called with named_parameters()
        """
        self.cfg.optimizer_fn = opt_fn

    def register_schedulers(
        self,
        scheduler_fn: Callable | None = None,
        lr_scheduler_config: None | dict = None,
    ):
        """Register a scheduler and/or scheduler config

        Parameters
        ----------
        scheduler_fn : Callable
            Function returning a Pytorch-compatible scheduler, taking the optimizer as
            the argument
        lr_scheduler_config : dict
            lr_scheduler_config as defined by Pytorch lightning
        """
        self.cfg.scheduler_fn = scheduler_fn
        self.cfg.scheduler_config = lr_scheduler_config

    @override
    def configure_optimizers(self):
        if self.cfg.optimizer_fn is not None:
            optimizer = self.cfg.optimizer_fn(self.named_parameters())
        else:
            optimizer = optim.Adam(self.named_parameters(), lr=0.001)
        lr_scheduler_config = (
            self.cfg.scheduler_config.copy()
            if self.cfg.scheduler_config is not None
            else {"monitor": "train_loss"}
        )
        if self.cfg.scheduler_fn is None:
            lr_scheduler_config["scheduler"] = schedule.ReduceLROnPlateau(
                optimizer=optimizer, patience=40
            )
        else:
            lr_scheduler_config["scheduler"] = self.cfg.scheduler_fn(optimizer)
        return optimizer

    @override
    def forward(self, X):
        "Should return unnormalized logits. predict_proba returns probabilities"
        raise NotImplementedError()

    def _score_regression(
        self, output: Tensor | tuple[Tensor], y_true: Tensor, prefix: str
    ) -> None:
        scores = self.metric_loggers[0](output, y_true)
        for name, score in zip(self.task_names, scores):
            self.log(f"{prefix}_mse_{name}", score)
        self._try_cache_to(f"{prefix}_mse", scores.mean())

    def _score_classification(
        self,
        output: Tensor | tuple[Tensor],
        y_true: Tensor,
        prefix: str,
    ) -> None:
        output = self._to_proba(output)
        if isinstance(output, tuple):  # Multiclass setting
            preds: Tensor = torch.hstack(
                [p.argmax(dim=1).reshape(-1, 1) for p in output]
            )
        else:
            preds = output.argmax(dim=1)
        if isinstance(output, tuple):
            accs = []
            for i, (name, y_true, pred) in enumerate(
                zip(self.task_names, self.iter_cols(y_true), self.iter_cols(preds))
            ):
                acc = self.metric_loggers[i](pred, y_true)
                accs.append(acc)
                self.log(f"{prefix}_acc_{name}", acc)
            self._try_cache_to(f"{prefix}_acc", torch.tensor(accs).mean())
        else:
            acc = self.metric_loggers[0](preds, y_true)
            self.log(f"{prefix}_acc_step", acc)
            self._try_cache_to(f"{prefix}_acc", acc)

    def predict_step(self, X):
        self.eval()
        if isinstance(X, Dataset):
            X = X[self.x_key][:]
        return self(X)

    def predict_proba(self, X) -> Tensor | tuple:
        self.eval()
        X = X[self.x_key][:]
        X = torch.tensor(X) if isinstance(X, np.ndarray) else X
        proba = self._to_proba(self(X))
        if isinstance(proba, tuple):
            return tuple(p.detach() for p in proba)
        return proba.detach()

    @override
    def training_step(self, batch, batch_idx):
        x = batch[self.x_key]
        if self.supervised and (
            self.task_type == "classification" or self.task_type == "regression"
        ):
            y: Tensor | None = torch.hstack(
                [batch[t].reshape(-1, 1) for t in self.task_names]
            )
        elif not self.supervised:
            y = None
        elif isinstance(self.task_names, str):
            y = batch[self.task_names]
        else:
            raise ValueError("no way to get y")
        output = self(x)
        loss = self.criterion(y_pred=output, y_true=y, batch=batch, context="train")
        self.log("train_loss", loss)
        self._try_cache_to("train_loss", loss)
        if self.cfg.record_norm:
            norm = (
                sum(
                    p.grad.norm(2).item() ** 2
                    for p in self.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            self.log("gradient_norm", norm)
        return loss

    @staticmethod
    def _to_proba(
        forward_out: tuple | Tensor, log: bool = False
    ) -> tuple[Tensor] | Tensor:
        "Convert logits from ``forward`` into normalized probabilities with softmax"
        if log:
            fn: Callable = nn.functional.log_softmax
        else:
            fn = nn.functional.softmax
        if isinstance(forward_out, tuple):
            return tuple([fn(p) for p in forward_out])
        return fn(forward_out)

    def _log_step(self, log_to, prefix: str, batch, batch_idx):
        x = batch[self.x_key]
        y = batch.get("y")
        output = self(x)
        loss = self.criterion(y_pred=output, y_true=y, context=prefix)
        self.log(log_to, loss)
        self._try_cache_to(log_to, loss)
        if self.cfg.record and self.task_type == "classification":
            self._score_classification(output=output, y_true=y, prefix=prefix)
        elif self.cfg.record and self.task_type == "regression":
            self._score_regression(output=output, y_true=y, prefix=prefix)
        return output

    @override
    def test_step(self, batch, batch_idx):
        _ = self._log_step("test_loss", "test", batch, batch_idx)

    @override
    def validation_step(self, batch, batch_idx):
        _ = self._log_step("val_loss", "val", batch, batch_idx)

    def reset_parameters(self):
        raise NotImplementedError()

    def criterion(
        self,
        y_pred,
        y_true,
        context: str | None = None,
        batch: dict | None = None,
        **kws,
    ):
        """criterion.

        Parameters
        ----------
        context : str
            Optional string denoting when the criterion was calculated e.g. "train"
            "validation" etc.
        """
        raise NotImplementedError()


# * MLP


class MLP(BaseNN):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: Callable = nn.ReLU,
        x_key: str = "x",
        batch_norm: bool = True,
        cfg: ModuleConfig | None = None,
    ):
        super().__init__(in_features=in_features, x_key=x_key, cfg=cfg)
        self.save_hyperparameters()
        if hidden_dim == -1:
            hidden_dim = in_features
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features=in_features, out_features=hidden_dim))
            layers.append(activation())
            if batch_norm:
                layers.append(nn.BatchNorm1d(num_features=hidden_dim))
            in_features = hidden_dim
        if self.task_type == "classification":
            self.outlayer: nn.ModuleList | None = nn.ModuleList(
                [nn.Linear(hidden_dim, n) for n in self.cfg.n_classes]
            )
        else:
            self.outlayer = None
            layers.append(nn.Linear(hidden_dim, self.n_tasks))
        self.nn = nn.Sequential(*layers)

    def criterion(
        self,
        y_pred,
        y_true,
        context: str | None = None,
        batch: dict | None = None,
        **kws,
    ):
        if self.cfg.task_type == "regression":
            multitask_all_reg(y_pred, y_true)
            loss_dict: dict = multitask_all_reg(
                y_pred, y_true, task_names=self.task_names, metrics=("mse",)
            )
            losses = torch.hstack([v["mse"] for _, v in loss_dict.items()])
            if self.cfg.task_weights:
                losses = losses * self.cfg.task_weights
            return losses.sum()
        return multitask_cross_entropy_loss(
            y_pred=y_pred,
            y_true=y_true,
            model=self if self.cfg.record else None,
            weights=self.cfg.task_weights,
        )

    def forward(self, X):
        X = X.to(torch.get_default_dtype())
        if self.task_type == "regression":
            return self.nn(X)
        if self.outlayer:
            hidden = self.nn(X)
            return tuple([o(hidden) for o in self.outlayer])
