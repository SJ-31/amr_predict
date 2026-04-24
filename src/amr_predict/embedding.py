#!/usr/bin/env ipython
from __future__ import annotations

import os
import uuid
from enum import Enum, EnumType
from pathlib import Path
from typing import ClassVar, Generator, Literal, override

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
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from loguru import logger
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
    if model == EmbeddingModels.esm3_open:
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
    _registry: ClassVar[dict[str, ModelEmbedder]] = {}
    model: EmbeddingModels
    token_amount: float | None = None
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
    save_proba: bool = False
    save_interval: int = 10
    pooling: BasicPoolings = BasicPoolings.MEAN
    only_cache: bool = True
    choose_hidden: bool = field(
        init=False, default=Factory(lambda self: self.hidden_layer < 0, takes_self=True)
    )
    valid_models: ClassVar[tuple[EmbeddingModels]] | None = None
    cache: EmbeddingCache = field(
        default=Factory(
            lambda self: EmbeddingCache(
                self.workdir,
                save_mode=self.save_mode,
                rng=self.rng,
                save_proba=self.save_proba,
                save_interval=self.save_interval,
                token_amount=self.token_amount,
                pooling=self.pooling,
            ),
            takes_self=True,
        )
    )

    @classmethod
    def __attrs_init_subclass__(cls):
        if cls.valid_models is not None:
            for model in cls.valid_models:
                ModelEmbedder._registry[model.name] = cls

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
    def new(_cls, model: EmbeddingModels, **kws):
        cls = ModelEmbedder._registry[model.name]
        return cls(**kws)

    def embed(
        self,
        dataset: Dataset | pl.DataFrame,
        text_key: str = "sequence",
        cols_remove: str | None = None,
    ) -> Dataset | None:
        df: pl.DataFrame = (
            dataset.to_polars() if not isinstance(dataset, pl.DataFrame) else dataset
        )
        self.cache.save(
            df[text_key], embed_fn=self._embed_batch, batch_size=self.batch_size
        )
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
    valid_models = tuple(EsmModels)
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
    valid_models = tuple(EsmSynthraModels)
    m: AutoModelForMaskedLM = field(
        init=False,
        default=Factory(
            lambda self: AutoModelForMaskedLM.from_pretrained(self.model.value),
            takes_self=True,
        ),
    )
    proba: TokenProbabilities = field(
        init=False, default=Factory(lambda self: TokenProbabilities(self.m.tokenizer))
    )

    def __attrs_post_init__(self):
        super().__attrs_post__init()
        self.m.to(self.device)

    def _embed_batch(
        self, sequences
    ) -> Generator[
        tuple[str, jaxtyping.Float[Tensor, "a b"], jaxtyping.Float[Tensor, "a"] | None]
    ]:
        if not self.choose_hidden:
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
    valid_models = tuple(EsmModels)
    tokenizer: EsmSequenceTokenizer = field(init=False, factory=EsmSequenceTokenizer)
    token2idx: dict[str, int] | None = field(
        init=False,
        default=Factory(  # 33 is the size of ESM's vocab
            lambda self: {self.tokenizer.decode(i): i for i in range(33)},
            takes_self=True,
        ),
    )

    @client.default
    def _get_client(self):
        if self.model == EsmModels.esm3_open:
            return ESM3.from_pretrained(self.model.name, device=self.device)
        else:
            return ESMC.from_pretrained(self.model.name)

    @lconf.default
    def _get_logits_conf(self):
        return LogitsConfig(
            return_embeddings=not self.choose_hidden,
            return_hidden_states=self.choose_hidden,
            sequence=True,
        )

    def _embed_batch(
        self, sequences
    ) -> Generator[
        tuple[str, jaxtyping.Float[Tensor, "a b"], jaxtyping.Float[Tensor, "a"] | None]
    ]:
        to_get = "hidden_states" if self.choose_hidden else "embeddings"
        for prot in sequences:
            encoded = self.client.encode(ESMProtein(sequence=prot))
            logit_out = self.client.logits(encoded, self.lconf)

            target: Tensor = getattr(logit_out, to_get)
            # Target has shape
            #    (n_hidden, 1, sequence_len, dim_model)
            # Otherwise
            #  (1, sequence_len, dim_model)
            # which is just the last hidden state by default
            if self.choose_hidden:
                tokens = target[self.hidden_layer, 0, 1:-1, :]
            else:
                tokens = target[0, 1:-1, :]
            logits: Tensor = logit_out.logits.sequence[0, :, :]
            logits = logits[1:-1, :]  # remove cls and eos tokens
            # shape of (1, sequence_len, vocab_size)
            all_proba = torch.softmax(logits, dim=1)
            proba = torch.tensor(
                [all_proba[i, self.token2idx[p]] for i, p in enumerate(prot)]
            )
            # NOTE: See https://github.com/evolutionaryscale/esm/issues/252
            # for how to convert logits to tokens
            yield prot, tokens, proba


@define
class SeqLensEmbedder(ModelEmbedder):
    model: EmbeddingModels = EmbeddingModels.seqLens_4096_512_46M_Mp
    valid_models = tuple(SeqLensModels)
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
    proba: TokenProbabilities = field(
        init=False,
        default=Factory(
            lambda self: TokenProbabilities(self.tokenizer), takes_self=True
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
                # Indices of tokens in the dictionary
                attention_mask = batch["attention_mask"].to(self.device)
                output: MaskedLMOutput = self.m(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                attention_mask = attention_mask.to("cpu")
                chosen_layer = output.hidden_states[self.hidden_layer]

                expanded_mask = attention_mask.unsqueeze(-1).expand_as(chosen_layer)
                hidden_layer = chosen_layer * expanded_mask

                for i, (seqlens_uid, input_id, logits, hidden) in enumerate(
                    zip(
                        torch.unbind(batch["seqlens_uid"]),
                        torch.unbind(batch["input_ids"]),
                        torch.unbind(output.logits),
                        torch.unbind(hidden_layer),
                    )
                ):
                    seq = seqlens_uid2seq[seqlens_uid.cpu().item()]
                    seq_proba = self.proba(input_id, logits, i == 0)
                    yield seq, hidden, seq_proba


@define
class TokenProbabilities:
    tokenizer: AutoTokenizer
    special: set = field(init=False)

    @special.default
    def _get_special_tokens(self):
        return set(self.tokenizer.special_tokens_map.values())

    def __call__(
        self,
        tokens: dict | Tensor,
        logits: jaxtyping.Float[Tensor, "a b"],
        check_compat: bool = False,
    ) -> Tensor:
        if check_compat:
            assert (
                self.tokenizer.vocab_size == logits.shape[1]
            ), "The tokenizer vocabulary size doesn't match that the logits dimension"
        probabilities = torch.softmax(logits, dim=1)
        iter = tokens["input_ids"] if isinstance(tokens, dict) else tokens
        seq_proba = [
            probabilities[i, id].repeat(len(decoded))
            for i, id in enumerate(iter)
            if (decoded := self.tokenizer.decode(id)) not in self.special
        ]
        return torch.concat(seq_proba)
