#!/usr/bin/env ipython

from typing import override

import lightning as L
from amr_predict.sae_external import BatchTopKSAE
from amr_predict.utils import ModuleConfig


class BatchTopK(L.LightningModule, BatchTopKSAE):
    def __init__(self, cfg: ModuleConfig, x_key: str = "embedding") -> None:
        super().__init__(cfg=cfg)
        self.x_key = x_key

    @override
    def training_step(self, batch, batch_idx):
        x = batch[self.x_key]
        output = super().forward(x)
        loss = output["loss"]
        # `loss` is the sum of l2_loss, l1_loss and aux_loss
        self.log("train_loss", loss)
        return loss
