#!/usr/bin/env ipython

from typing import override

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from amr_predict.models import BaseNN
from amr_predict.sae_external import BatchTopKSAE
from amr_predict.utils import ModuleConfig
from torch import Tensor

# References
# [1] https://github.com/decoderesearch/SAELens/blob/main/sae_lens/saes/jumprelu_sae.py
#

# TODO: looks like this has no dead feature resampling
#
# TODO: no bias initialization, but newer tutorials don't mention this as being necessary
# i.e. "initializing decoder bias with geometric median of activations"


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
        self.threshold: Tensor = torch.tensor(0, dtype=self.dtype, device=self.device)

    @override
    def predict_step(self, X: Tensor):
        """For inference
        Return SAE activations for a batch of samples i.e. ReLU((X+b)*W_e)
        """
        if X.shape[1] != self.in_features:
            raise ValueError(
                f"The sample dimensions ({X.shape[0]}) don't match what the model was trained on {self.in_features}"
            )
        with torch.no_grad():
            X, _, _ = self.m.preprocess_input(X)
            batch_cent = X - self.m.b_dec
            # JumpReLU activation function to remove batch dependency
            acts = F.relu(batch_cent @ self.m.W_enc)
            relu_mask = (acts > self.threshold).to(acts.dtype)
            return acts * relu_mask

    def update_threshold(self, topk_acts: Tensor):
        learning_rate = self.cfg.get("top_k_threshold_lr", 0.01)
        positive_mask = topk_acts > 0
        if positive_mask.any():
            min_positive = topk_acts[positive_mask].min().to(self.threshold.dtype)
            self.threshold = (
                1 - learning_rate
            ) * self.threshold + learning_rate * min_positive

    @override
    def training_step(self, batch, batch_idx):
        x = batch[self.x_key]
        output = self.m.forward(x)
        loss = output["loss"]
        # `loss` is the sum of l2_loss, l1_loss and aux_loss
        self.log("train_loss", loss)
        self._try_cache_to("train_loss", loss)
        self.update_threshold(output["feature_acts"])
        self.log_dict(
            {
                k: v
                for k, v in output.items()
                if k not in {"sae_out", "loss", "feature_acts"}
            }
        )
        return loss
