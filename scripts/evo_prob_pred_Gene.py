# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Literal, Optional

import nemo.lightning as nl
import numpy as np
import pandas as pd
import torch
from bionemo.evo2.data.fasta_dataset import SimpleFastaDataset
from bionemo.llm.lightning import LightningPassthroughPredictionMixin
from bionemo.llm.utils.callbacks import PredictionWriter
from lightning.pytorch import LightningDataModule
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import _gather_along_last_dim
from megatron.core.utils import get_batch_on_this_cp_rank
from nemo.collections.llm.gpt.model.base import get_packed_seq_params
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS, HyenaModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.data import WrappedDataLoader
from torch import Tensor

logging.getLogger("nemo_logger").setLevel(logging.ERROR)

CheckpointFormats = Literal["torch_dist", "zarr"]


for i in range(torch.cuda.device_count()):
    print(f"Logical GPU {i}: {torch.cuda.get_device_name(i)}")


def _gather_along_cp_dim(input_, seq_dim: int = 1):
    """Gather tensors and concatenate along the last dimension."""
    world_size = parallel_state.get_context_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )
    torch.distributed.all_gather_into_tensor(
        output,
        input_.contiguous(),
        group=parallel_state.get_tensor_model_parallel_group(),
    )
    tensor_list = output.chunk(world_size, dim=0)
    output = torch.cat(tensor_list, dim=seq_dim).contiguous()

    return output


class HyenaPredictor(LightningPassthroughPredictionMixin, HyenaModel):
    """A predictor for the Hyena model. This adds in the predict step and the passthrough method."""

    def __init__(
        self,
        *args,
        output_log_prob_seqs: bool = False,
        log_prob_collapse_option: Literal["sum", "mean"] = "mean",
        **kwargs,
    ):
        """Initialize the predictor with our needs around computing log probabilities."""
        super().__init__(*args, **kwargs)
        self.output_log_prob_seqs = output_log_prob_seqs
        self.log_prob_collapse_option = log_prob_collapse_option

    def predict_step(self, batch, batch_idx: Optional[int] = None) -> Tensor:
        """Alias for forward_step, also log the pad mask since sequences may not all have the same length."""
        if len(batch) == 0:
            return
        token_numpy = batch["tokens"].cpu().numpy().reshape(-1, 1)
        # print(token_numpy.shape)
        # print(self.tokenizer.ids_to_text(token_numpy[-1]))
        # print(self.tokenizer.ids_to_tokens(token_numpy[-1]))
        # print(token_numpy[-1], token_numpy[-2])

        forward_out, embedding_out = self.forward_step(batch)

        if not isinstance(forward_out, Tensor) and not isinstance(
            embedding_out, Tensor
        ):
            return forward_out_gathered

        # Reminder: the model's predictions for input i land at output i+1. To get everything to align, we prepend the
        # EOS token to the input sequences and take the outputs for all but the first token.
        forward_out_tp_gathered = _gather_along_last_dim(
            forward_out, group=parallel_state.get_tensor_model_parallel_group()
        )
        # print(forward_out_tp_gathered.shape)
        # else:
        #     forward_out_tp_gathered = _collect_into_dim(forward_out, dim=-1)
        forward_out_gathered = _gather_along_cp_dim(forward_out_tp_gathered)
        # print("self.tokenizer.vocab_size", self.tokenizer.vocab_size)
        assert self.tokenizer.vocab_size == forward_out_gathered.shape[-1]
        if self.output_log_prob_seqs:
            softmax_logprobs = torch.log_softmax(forward_out_gathered, dim=-1)
            softmax_logprobs = softmax_logprobs[:, :-1]
            input_ids = batch["tokens"][:, 1:]
            assert softmax_logprobs.shape[1] == input_ids.shape[1]

            logprobs = torch.gather(
                softmax_logprobs,  # Gather likelihoods...
                2,  # along the vocab dimension...
                input_ids.unsqueeze(-1),  # using the token ids to index.
            ).squeeze(-1)
            log_prob_seqs = torch.sum(
                logprobs * batch["loss_mask"][:, 1:].float(), dim=-1
            )
            if self.log_prob_collapse_option == "mean":
                log_prob_seqs = log_prob_seqs / (
                    batch["loss_mask"][:, 1:].float().sum(dim=-1) + 1e-8
                )

            return {
                "log_probs_seqs": log_prob_seqs.cpu(),
                "seq_idx": batch["seq_idx"].cpu(),
                "embedding": embedding_out,
                "sequence_length": batch["loss_mask"][:, 1:].float().sum(dim=-1).cpu(),
            }
        else:
            # If the user wants to match back to logits, then they will need to do the offsetting logic themselves.
            return {
                "token_logits": forward_out_gathered.cpu(),
                "pad_mask": batch["loss_mask"].cpu(),
                "seq_idx": batch["seq_idx"].cpu(),
                "embedding": embedding_out,
            }


def hyena_predict_forward_step(model, batch) -> torch.Tensor:
    """Performs a forward step for the Hyena model.

    Args:
        model: The Hyena model
        batch: Dictionary containing input batch data with keys:
            - tokens: Input token IDs
            - position_ids: Position IDs
            - labels: Labels for loss computation
            - loss_mask: Mask for loss computation

    Returns:
        torch.Tensor: Output from the model forward pass
    """

    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        # "labels": batch["labels"],
        # "loss_mask": batch["loss_mask"],
    }

    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output[0].detach()

        return hook

    forward_args["attention_mask"] = None
    if "cu_seqlens" in batch:
        forward_args["packed_seq_params"] = get_packed_seq_params(batch)

    # Choose layer for embedding
    model._modules["module"].module.module.decoder.layers[
        30
    ].mlp.linear_fc2.register_forward_hook(get_activation("embedding_output"))

    model_output = model(**forward_args)

    avg_embedding_output = torch.sum(activations["embedding_output"], axis=0)

    return model_output, avg_embedding_output


def hyena_predict_data_step(dataloader_iter) -> dict[str, torch.Tensor]:
    """Data step for the Hyena model prediction. Modified from the original gpt data step to include the seq_idx."""
    from megatron.core import parallel_state

    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842

    batch = next(dataloader_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_device_keys = set()
    required_host_keys = set()

    required_device_keys.add("attention_mask")
    if "cu_seqlens" in _batch:
        required_device_keys.add("cu_seqlens")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")

    if parallel_state.is_pipeline_first_stage():
        required_device_keys.update(("tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_device_keys.update(("labels", "loss_mask", "seq_idx"))

    _batch_required_keys = {}
    for key, val in _batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True)
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu()
        else:
            _batch_required_keys[key] = None

    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_cp_rank(_batch_required_keys)

    return output


class PredictDataModule(LightningDataModule):
    """Create a dataloader for prediction."""

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int = 1):
        """Create a dataloader for prediction."""
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the dataloader."""
        pass

    def predict_dataloader(self):
        """Create a dataloader for prediction."""
        # need to use this to communicate that we are in predict mode and safe to not drop last batch
        return WrappedDataLoader(
            mode="predict",
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
        )


def predict(
    fasta_path: Path,
    ckpt_dir: str,
    output_dir: Path,
    json_output_path: Path,
    parquet_output_path: Path,
    tensor_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    model_size: str = "7b",
    ckpt_format: CheckpointFormats = "torch_dist",
    fp8: bool = False,
    full_fp8: bool = False,
    work_dir: Path | None = None,
    batch_size: int = 1,
    output_log_prob_seqs: bool = False,
    log_prob_collapse_option: Literal["sum", "mean"] = "mean",
    prepend_bos: bool = False,
    no_sequence_parallel: bool = False,
    hybrid_override_pattern: str | None = None,
    num_layers: int | None = None,
):
    """Inference workflow for Evo2.

    Returns:
        None
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())
    sequence_parallel = tensor_parallel_size > 1 and not no_sequence_parallel
    output_dir.mkdir(
        parents=True, exist_ok=True
    )  # Make sure the output directory exists, files will be written here.
    model_parallel_size = (
        tensor_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )
    if model_parallel_size > torch.cuda.device_count():
        raise ValueError(
            f"Requested model parallel size {model_parallel_size} is greater than the "
            f"number of available CUDA devices {torch.cuda.device_count()}"
        )

    # Create PTL trainer.
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=model_parallel_size,
        strategy=nl.MegatronStrategy(
            drop_last_batch=False,
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,  # Needs to be false for a normal model checkpoint.
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            sequence_parallel=tensor_parallel_size > 1 and sequence_parallel,
            save_ckpt_format=ckpt_format,
            ckpt_load_strictness="log_all",
            data_sampler=nl.MegatronDataSampler(
                micro_batch_size=batch_size,
                global_batch_size=batch_size,
                seq_len=8192,
                output_log=False,  # this is needed for predict step to work
            ),
        ),
        log_every_n_steps=1,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        callbacks=[
            PredictionWriter(
                output_dir=output_dir,
                write_interval="epoch",
                batch_dim_key_defaults={"token_logits": 0},
                seq_dim_key_defaults={"token_logits": 1},
            )
        ],
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            # Only use FP8 in this plugin when using full FP8 precision and FP8.
            #   Otherwise use vortex_style_fp8 in the model config.
            fp8="hybrid" if fp8 and full_fp8 else None,
            fp8_amax_history_len=16 if fp8 and full_fp8 else 1,
            fp8_amax_compute_algo="max" if fp8 and full_fp8 else "most_recent",
        ),
    )
    # The following two config options are really only used for testing, but may also be useful for getting output from
    #   specific layers of the model.
    config_modifiers_init = {}
    if hybrid_override_pattern is not None:
        config_modifiers_init["hybrid_override_pattern"] = hybrid_override_pattern
    if num_layers is not None:
        config_modifiers_init["num_layers"] = num_layers

    config = HYENA_MODEL_OPTIONS[model_size](
        forward_step_fn=hyena_predict_forward_step,
        data_step_fn=hyena_predict_data_step,  # , attention_backend=AttnBackend.fused,
        distribute_saved_activations=False
        if sequence_parallel and tensor_parallel_size > 1
        else True,
        # Only use vortex style FP8 in the model config if using FP8 and not full FP8. This will only apply FP8 to
        #   the projection layer of the hyena mixer.
        vortex_style_fp8=fp8 and not full_fp8,
        **config_modifiers_init,
    )
    trainer.strategy._setup_optimizers = False

    nemo_logger = NeMoLogger(log_dir=work_dir)
    nemo_logger.setup(trainer, resume_if_exists=True)
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=False,
        resume_past_end=False,
        restore_config=nl.RestoreConfig(
            path=str(ckpt_dir),  # NeMo expects a string path.
            load_model_state=True,
            load_optim_state=False,
            load_artifacts=False,
        ),
    )
    tokenizer = get_nmt_tokenizer("byte-level")
    model = HyenaPredictor(
        config,
        tokenizer=tokenizer,
        output_log_prob_seqs=output_log_prob_seqs,
        log_prob_collapse_option=log_prob_collapse_option,
    )

    print(json_output_path)
    print("Set up model")
    resume.setup(trainer, model)  # this pulls weights from the starting checkpoint.

    print("Create FASTA dataset")
    dataset = SimpleFastaDataset(fasta_path, tokenizer, prepend_bos=prepend_bos)

    datamodule = PredictDataModule(dataset, batch_size=batch_size)

    print("Predict sequence")

    if torch.cuda.is_available():
        # Get the current GPU device
        print("Free Mem", torch.cuda.mem_get_info()[0])
        print("Total Mem", torch.cuda.mem_get_info()[1])

    prediction = trainer.predict(model, datamodule=datamodule)

    processed_prediction = None
    sequence_length = 0
    all_embeddings = []
    ids = []
    for i in range(len(prediction)):
        pred = prediction[i]["embedding"].float().detach().cpu().numpy()
        if i == 0:
            processed_prediction = pred
        else:
            processed_prediction += pred
        all_embeddings.append(pred[0])
        ids.append(prediction[i]["seq_idx"].numpy()[0])

        sequence_length += (
            prediction[i]["sequence_length"].float().detach().cpu().numpy().tolist()[0]
        )

    processed_prediction = processed_prediction / sequence_length

    prediction_json = {
        "avg_embedding": processed_prediction.tolist(),
        "sequence_length": sequence_length,
    }
    edf: pd.DataFrame = pd.DataFrame(np.array(all_embeddings)).assign(id=ids)

    with open(json_output_path, "w") as json_file:
        json.dump(prediction_json, json_file)
    edf.to_parquet(parquet_output_path, index=False)

    print("Write files")
    dataset.write_idx_map(
        output_dir
    )  # Finally write out the index map so we can match the predictions to the original sequences.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint_path", required=True)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--outdir", required=True)
    args = vars(parser.parse_args())  # convert to dict
    return args


if __name__ == "__main__":
    args = parse_args()
    input: Path = Path(args["input"])
    outdir: Path = Path(args["outdir"])
    checkpoint_path: Path = Path(args["checkpoint_path"])
    if not checkpoint_path.exists():
        raise ValueError("Model checkpoint not found")
    extensions = {".fasta", ".fa", ".fna"}
    if input.is_dir():
        for file in input.iterdir():
            if file.suffix in extensions:
                json_out = outdir / f"{input.stem}.json"
                parquet_out = outdir / f"{input.stem}.parquet"
                predict(
                    fasta_path=Path(file),
                    ckpt_dir=checkpoint_path,
                    tensor_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    context_parallel_size=1,
                    batch_size=1,
                    output_dir=outdir,
                    json_output_path=json_out,
                    parquet_output_path=parquet_out,
                    model_size="7b",
                    ckpt_format="torch_dist",
                    prepend_bos=True,
                    output_log_prob_seqs=True,
                    log_prob_collapse_option="mean",
                )
    else:
        json_out = outdir / f"{input.stem}.json"
        parquet_out = outdir / f"{input.stem}.parquet"
        predict(
            fasta_path=input,
            ckpt_dir=checkpoint_path,
            tensor_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            batch_size=1,
            output_dir=outdir,
            json_output_path=json_out,
            parquet_output_path=parquet_out,
            model_size="7b",
            ckpt_format="torch_dist",
            prepend_bos=True,
            output_log_prob_seqs=True,
            log_prob_collapse_option="mean",
        )
