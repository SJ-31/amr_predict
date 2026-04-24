#!/usr/bin/env ipython
from __future__ import annotations

import os
import uuid
from enum import Enum, EnumType
from pathlib import Path
from typing import Generator, Literal, override

import jaxtyping
import polars as pl
import torch
from amr_predict.cache import EmbeddingCache
from amr_predict.pooling import BasicPoolings
from attrs import Factory, define, field, validators
from datasets.arrow_dataset import Dataset
from esm.models.esm3 import ESM3
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding
from transformers.modeling_outputs import MaskedLMOutput

EsmSynthraModels = Enum(
    "EsmSynthraModels",
    {
        "esmc_600m_synthyra": "Synthyra/ESMplusplus_large",
        "esmc_300m_synthyra": "Synthyra/ESMplusplus_small",
    },
)


EsmModels = Enum(
    "EsmModels",
    {
        i: i
        for i in [
            "esm2_t6_8m_UR50D",
            "esm2_t33_650m_UR50D",
            "esm3_open",
            "esmc_600m",
            "esmc_300m",
        ]
    },
)


SeqLensModels = Enum(
    "SeqLensModels", {"seqLens_4096_512_46M_Mp": "omicseye/seqLens_4096_512_46M-Mp"}
)


def embedding_size(model: EmbeddingModels) -> int:
    if isinstance(model, EmbeddingModels.esm3_open):
        raise NotImplementedError("figure this out")
    elif validate_model_group(model, SeqLensModels):
        return 512
    elif validate_model_group(model, EsmSynthraModels) or validate_model_group(
        model, EsmModels
    ):
        return 2048
    raise NotImplementedError()


EmbeddingModels = Enum(
    "EmbeddingModels",
    dict(
        {i.name: i.value for i in EsmSynthraModels}
        | {i.name: i.value for i in SeqLensModels}
        | {i.name: i.value for i in EsmModels}
    ),
)


def validate_model_group(model: EmbeddingModels, group: EnumType) -> bool:
    try:
        group[model.name]
        return True
    except KeyError:
        return False


@define
class ModelEmbedder:
    model: EmbeddingModels
    token_prop: float | None = None
    workdir: Path = field(
        factory=lambda: Path().cwd().joinpath(f"embedding_work_{uuid.uuid4().hex}")
    )
    save_mode: Literal["tokens", "seqs", "both"] = field(
        default="seqs", validator=validators.in_(("tokens", "seqs", "both"))
    )
    default_dtype: torch.dtype = torch.float32
    device: str = field(factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 128
    hidden_layer: int = 0
    huggingface: str | None = None
    rng: int | None = None
    save_logits: bool = False
    save_interval: int = 10
    pooling: BasicPoolings = BasicPoolings.MEAN
    only_cache: bool = True
    get_hidden: bool = field(
        init=False, default=Factory(lambda self: self.hidden_layer < 0, takes_self=True)
    )
    cache: EmbeddingCache = field(
        default=Factory(
            lambda self: EmbeddingCache(
                self.workdir,
                save_mode=self.save_mode,
                rng=self.rng,
                save_logits=self.save_logits,
                save_interval=self.save_interval,
                token_prop=self.token_prop,
                pooling=self.pooling,
            ),
            takes_self=True,
        )
    )

    def __attrs_post_init__(self):
        if self.huggingface:
            os.environ["HF_HOME"] = self.huggingface
        if not self.workdir.exists():
            self.workdir.mkdir()
        self.hidden_layer = 0 if self.hidden_layer < 0 else self.hidden_layer

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

    @classmethod
    def new(
        _cls,
        model: EmbeddingModels,
        save_mode: Literal["tokens", "seqs", "both"] = "seqs",
        default_dtype: torch.dtype = torch.float32,
        batch_size: int = 128,
        workdir: Path | None = None,
        hidden_layer: int = 0,
        huggingface: str | None = None,
        save_logits: bool = False,
        save_interval: int = 10,
        only_cache: bool = True,
    ):
        cls: ModelEmbedder
        if validate_model_group(model, EsmSynthraModels):
            cls = EsmSynthyra
        elif validate_model_group(model, SeqLensModels):
            cls = SeqLensEmbedder
        elif validate_model_group(model, EsmModels):
            cls = EsmOfficial
        else:
            raise NotImplementedError()
        return cls(
            save_mode=save_mode,
            workdir=workdir,
            default_dtype=default_dtype,
            batch_size=batch_size,
            hidden_layer=hidden_layer,
            huggingface=huggingface,
            save_logits=save_logits,
            save_interval=save_interval,
            only_cache=only_cache,
        )

    def embed(
        self,
        dataset: Dataset | pl.DataFrame,
        text_key: str = "sequence",
        cols_remove: str | None = None,
    ) -> Dataset | None:
        df: pl.DataFrame = (
            dataset.to_polars() if not isinstance(dataset, pl.DataFrame) else dataset
        )
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
            yield prot, tokens.to("cpu"), Tensor()


@define
class EsmSynthyra(ModelEmbedder):
    model: EmbeddingModels = field(default=EsmSynthraModels.esmc_600m_synthyra)

    def __attrs_post_init__(self):
        super().__attrs_post__init()
        key: str = self.model.value
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
                    yield prot, tokens.to("cpu"), Tensor()


@define
class EsmOfficial(ModelEmbedder):
    model: EmbeddingModels = field(default=EsmModels.esmc_600m)
    client: ESM3 | ESMC = field(init=False)
    lconf: LogitsConfig = field(init=False)

    @client.default
    def _get_client(self):
        if self.model == EsmModels.esm3_open:
            return ESM3.from_pretrained(self.model.name, device=self.device)
        else:
            return ESMC.from_pretrained(self.model.name)

    @lconf.default
    def _get_logits_conf(self):
        return LogitsConfig(
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


@define
class SeqLensEmbedder(ModelEmbedder):
    model: EmbeddingModels = EmbeddingModels.seqLens_4096_512_46M_Mp
    tokenizer: AutoTokenizer = field(
        init=False,
        default=Factory(
            lambda self: AutoTokenizer.from_pretrained(self.model.value),
            takes_self=True,
        ),
    )
    m: AutoModelForMaskedLM = field(
        init=False,
        default=Factory(
            lambda self: AutoModelForMaskedLM.from_pretrained(self.model.value),
            takes_self=True,
        ),
    )

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.m.to(self.device)
        self.m.config.output_hidden_states = True
        torch.set_default_dtype(self.default_dtype)

    def _embed_batch(
        self, sequences
    ) -> Generator[
        tuple[str, jaxtyping.Float[Tensor, "a b"], jaxtyping.Float[Tensor, "a"] | None]
    ]:
        dataset = Dataset.from_dict({"sequence": list(sequences)})
        dataset = dataset.add_column("seqlens_uid", list(range(len(dataset))))
        to_remove = [c for c in dataset.column_names if c != "seqlens_uid"]
        seqlens_uid2seq = dict(zip(dataset["seqlens_uid"], dataset["sequence"]))

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        tokenized = dataset.map(
            lambda x: self._tokenize(
                x, tokenizer=self.tokenizer, text_key="sequence", max_length=512
            ),
            batched=True,
            remove_columns=to_remove,
        )
        loader = DataLoader(
            tokenized, batch_size=self.batch_size, collate_fn=data_collator
        )

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                output: MaskedLMOutput = self.m(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                attention_mask = attention_mask.to("cpu")
                chosen_layer = output.hidden_states[self.hidden_layer]

                expanded_mask = attention_mask.unsqueeze(-1).expand_as(chosen_layer)
                hidden_layer = chosen_layer * expanded_mask

                for i, seqlens_uid in enumerate(torch.unbind(batch["seqlens_uid"])):
                    seq = seqlens_uid2seq[seqlens_uid.cpu().item()]
                    token = hidden_layer[i, :, :]
                    yield seq, token, Tensor()
