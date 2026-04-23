#!/usr/bin/env ipython

#!/usr/bin/env ipython
from __future__ import annotations

import os
import uuid
from enum import Enum
from pathlib import Path
from typing import Callable, Generator, Literal, get_args, override

import jaxtyping
import polars as pl
import torch
from amr_predict.pooling import BASIC_POOLING_METHODS, pool_tensor
from amr_predict.utils import EmbeddingCache
from attrs import Factory, define, field, validators
from datasets.arrow_dataset import Dataset
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding
from transformers.modeling_outputs import MaskedLMOutput

EsmModels = Enum(
    "EsmModels",
    [
        "esm2_t6_8m_UR50D",
        "esm2_t33_650m_UR50D",
        "esm3_open",
        "esmc_600m",
        "esmc_300m",
        "esmc_600m_synthyra",
        "esmc_300m_synthyra",
    ],
)

EmbeddingModels = Enum(
    "EmbeddingMethods",
    ["Evo2", "seqLens"] + [n.name for n in EsmModels],
)


@define
class ModelEmbedder:
    method: EmbeddingModels
    token_prop: float | None = None
    workdir: Path = field(
        factory=Factory(
            lambda self: Path()
            .cwd()
            .joinpath(f"{self.method.name}-work_{uuid.uuid4().hex}"),
            takes_self=True,
        )
    )
    save_mode: Literal["tokens", "seqs", "both"] = "seqs"
    default_dtype: torch.dtype = torch.float32
    device: str = field(
        factory=Factory(lambda: "cuda" if torch.cuda.is_available() else "cpu")
    )
    hidden_layer: int = 0
    huggingface: str | None = None
    save_proba: bool = False
    save_interval: int = 10
    only_cache: bool = True

    def __attrs_post_init__(self):
        os.environ["HF_HOME"] = self.huggingface
        if not self.workdir.exists():
            self.workdir.mkdir()
        self.cache: EmbeddingCache = EmbeddingCache(
            self.workdir,
            save_mode=self.save_mode,
            save_interval=self.save_interval,
            token_prop=self.token_prop,
        )

    @property
    def get_hidden(self) -> bool:
        return self.hidden_layer < 0

    @property
    def with_tokens(self) -> bool:
        return self.save_mode != "seqs"

    def _embed_batch(
        self, sequences
    ) -> Generator[
        tuple[str, jaxtyping.Float[Tensor, "a b"], jaxtyping.Float[Tensor, "a"] | None]
    ]:
        """Generic function that embeds a batch of sequences for use with EmbeddingCache.
        Returns a tuple of the embedded sequence, tensor of embedded tokens, and optionally
        tensor of token probabilities
        """
        raise NotImplementedError()

    def embed(
        self,
        dataset: Dataset | None,
        text_key: str = "sequence",
        cols_remove: str | None = None,
    ) -> Dataset | None:
        df: pl.DataFrame = dataset.to_polars()
        self.cache.save(df[text_key], fn=self._embed_batch, batch_size=self.batch_size)
        if not self.only_cache:
            result = self.cache.to_dataset(df, key_col=text_key)
            if cols_remove is not None:
                result = result.remove_columns(cols_remove)
            return result

    @staticmethod
    def _tokenize(data, tokenizer, text_key: str, max_length: int):
        return tokenizer(
            data[text_key],
            truncation=True,
            padding=False,
            max_length=max_length,
        )


@define
class Esm2Embedder(ModelEmbedder):
    model_name: EsmModels = field(
        default=EsmModels.esm2_t33_650m_UR50D,
        validator=lambda inst, cls, val: val.name.startswith("esm2"),
    )

    @override
    def _embed_batch(
        self, sequences
    ) -> Generator[
        tuple[str, jaxtyping.Float[Tensor, "a b"], jaxtyping.Float[Tensor, "a"] | None]
    ]:
        from interplm.esm.embed import embed_single_sequence

        for prot in sequences:
            token_embeddings = embed_single_sequence(
                prot, model_name=self.model_name.name, layer=self.hidden_layer
            )
            tokens: Tensor = token_embeddings if self.with_tokens else None
            yield prot, tokens, Tensor()


@define
class EsmSynthyra(ModelEmbedder):
    model: EmbeddingModels = field(
        default=EsmModels.esmc_600m_synthyra,
        validator=validators.or_(
            validators.instance_of(EsmModels.esmc_600m_synthyra),
            validators.instance_of(EsmModels.esmc_300m_synthyra),
        ),
    )

    def __attrs_post_init__(self):
        key: dict[EsmModels, str] = {
            EsmModels.esmc_600m_synthyra: "Synthyra/ESMplusplus_large",
            EsmModels.esmc_300m_synthyra: "Synthyra/ESMplusplus_small",
        }[self.model]
        self.m: AutoModelForMaskedLM = AutoModelForMaskedLM.from_pretrained(
            key, trust_remote_code=True
        )
        self.m.to(self.device)

    def _embed_batch(
        self, sequences
    ) -> Generator[
        tuple[str, jaxtyping.Float[Tensor, "a b"], jaxtyping.Float[Tensor, "a"] | None]
    ]:
        if not self.get_hidden:
            tmp = self.m.embed_dataset(
                tokenizer=self.m.tokenizer,
                sequences=sequences,
                batch_size=self.batch_size,
                full_embeddings=True,
                max_len=2048,
                pooling_types=[],
                save=False,
            )
            for k, v in tmp.items():
                yield k, v, Tensor()
        else:
            with torch.no_grad():
                for prot in sequences:
                    tokenized = self.m.tokenizer(
                        prot, padding=False, return_tensors="pt"
                    ).to(self.device)
                    output = self.m(**tokenized, output_hidden_states=True)
                    hidden = output.hidden_states
                    # Each element of hidden is a tensor with shape
                    # (1, sequence_len, dim_model)
                    tokens = hidden[self.hidden_layer][0, 1:-1, :]
                    yield prot, tokens, Tensor()


@define
class EsmOfficial(ModelEmbedder):
    model: EmbeddingModels = field(
        default=EsmModels.esmc_600m,
        validator=validators.or_(
            validators.instance_of(EsmModels.esmc_300m),
            validators.instance_of(EsmModels.esmc_600m),
            validators.instance_of(EsmModels.esm3_open),
        ),
    )

    def __attrs_post_init__(self):
        from esm.sdk.api import LogitsConfig

        if isinstance(self.model, EsmModels.esm3_open):
            from esm.models.esm3 import ESM3

            self.client = ESM3.from_pretrained(self.model.name, device=self.device)
        else:
            from esm.models.esmc import ESMC

            self.client = ESMC.from_pretrained(self.model.name)

        self.lconf = LogitsConfig(
            return_embeddings=not self.get_hidden, return_hidden_states=self.get_hidden
        )

    def _embed_batch(
        self, sequences
    ) -> Generator[
        tuple[str, jaxtyping.Float[Tensor, "a b"], jaxtyping.Float[Tensor, "a"] | None]
    ]:
        to_get = "hidden_states" if self.get_hidden else "embeddings"
        for prot in sequences:
            encoded = self.client.encode(ESMProtein(sequence=prot))
            logits = self.client.logits(encoded, self.lconf)

            target: Tensor = getattr(logits, to_get)
            # If `get_hidden`, target has shape
            #    (n_hidden, 1, sequence_len, dim_model)
            # Otherwise
            #  (1, sequence_len, dim_model)
            if self.get_hidden:
                tokens = target[self.hidden_layer, 0, 1:-1, :]
            else:
                tokens = target[0, 1:-1, :]
            yield prot, tokens, Tensor()
