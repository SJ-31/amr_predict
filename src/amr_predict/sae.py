#!/usr/bin/env ipython

from typing import override

import lightning as L
from amr_predict.models import BaseNN
from amr_predict.sae_external import BatchTopKSAE
from amr_predict.utils import ModuleConfig


class BatchTopK(BaseNN):
    def __init__(self, cfg: ModuleConfig, x_key: str = "embedding") -> None:
        # Relevant keys in cfg are
        # - top_k
        # - l1_coeff
        # - n_batches_to_dead
        # - seed
        # - act_size # this is the size of the input
        # - dict_size # number of latents
        # - input_unit_norm
        super().__init__(in_features=cfg["act_size"], x_key=x_key, cfg=cfg)
        self.n_tasks = 1
        self.task_names = []
        self.task_type = "reconstruction"
        self.n_classes = 1
        self.m = BatchTopKSAE(cfg)
        self.task_names = x_key

    @override
    def training_step(self, batch, batch_idx):
        x = batch[self.x_key]
        output = self.m.forward(x)
        loss = output["loss"]
        # `loss` is the sum of l2_loss, l1_loss and aux_loss
        self.log("train_loss", loss)
        self._try_cache_to("train_loss", loss)
        for k, v in output.items():
            if k in {"sae_out", "loss", "feature_acts"}:
                continue
            self.log(k, v)
        return loss
