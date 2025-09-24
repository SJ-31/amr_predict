#!/usr/bin/env ipython
from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import polars as pl
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from datasets import Array2D, Features, concatenate_datasets
from datasets.arrow_dataset import Dataset
from datasets.load import load_from_disk
from polars.exceptions import NoRowsReturnedError
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding
from transformers.modeling_outputs import MaskedLMOutput

SPLIT_METHODS: TypeAlias = Literal["bin", "bakta"]
EMBEDDING_METHODS: TypeAlias = Literal["seqLens", "Evo2"]


def add_intergenic(
    record: SeqRecord,
    df: pl.DataFrame,
    start_col: str = "Start",
    stop_col: str = "Stop",
) -> pl.DataFrame:
    final_intergenic = len(record) - max(df[stop_col])
    df = df.with_columns(
        downstream_intergenic=np.concat(
            [df["Start"][1:] - df["Stop"][:-1], [final_intergenic]]
        )
    )
    first_intergenic = min(df[start_col])
    df = df.with_columns(
        upstream_intergenic=np.concat(
            [[first_intergenic], df["downstream_intergenic"][:-1]]
        )
    )
    return df


class SeqEmbedder:
    def __init__(
        self,
        method: EMBEDDING_METHODS = "seqLens",
        tokenizer: AutoTokenizer | None = None,
        **kwargs,
    ):
        self.method: EMBEDDING_METHODS = method
        self.tokenizer: AutoTokenizer | None = tokenizer
        self.kwargs: dict = kwargs

    def __call__(self, dataset: Dataset | SeqDataset) -> Dataset:
        if self.method == "seqLens":
            return self._seqlens_embed(dataset, **self.kwargs)
        elif self.method == "Evo2":
            raise ValueError("Not implemented yet")
        else:
            raise ValueError("Given method not supported!")

    def _seqlens_embed(
        self,
        dataset: Dataset,
        huggingface: str,
        text_key: str = "sequence",
        model_key: str = "omicseye/seqLens_4096_512_46M-Mp",
        batch_size: int = 64,
        pooling: Literal["mean", "cls", "max", "concat"] = "mean",
    ) -> Dataset:
        """Generate embeddings of `dataset` with seqLens

        Parameters
        ----------
        huggingface : str
            Path to huggingface cache on disk
        pooling : Literal
            Method for pooling sequence hidden states. Current implementations
            are from the seqLens repository
        text_key : str
            Column name of `dataset` containing the sequences
        """
        os.environ["HF_HOME"] = huggingface
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_key)
        model = AutoModelForMaskedLM.from_pretrained(model_key)
        model.config.output_hidden_states = True
        to_keep = {"_index", "label"}
        dataset = dataset.add_column("_index", list(range(len(dataset))))
        to_remove = [c for c in dataset.column_names if c not in to_keep]
        tokenized = dataset.map(
            lambda x: self._tokenize_fn(x, text_key=text_key, max_length=512),
            batched=True,
            remove_columns=to_remove,
        )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # TODO: is there any reason to change this from the default?
        loader = DataLoader(tokenized, batch_size=batch_size, collate_fn=data_collator)
        hidden_size = []

        def gen():
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    output: MaskedLMOutput = model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    last_hidden = output.hidden_states[-1]
                    # This has shape (batch_size, sequence_length, hidden_size)
                    # Mask out padding
                    expanded_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden)
                    hidden_masked = last_hidden * expanded_mask
                    if pooling == "max":
                        embedding = hidden_masked.max(dim=1).values
                    elif pooling == "cls":
                        embedding = hidden_masked[:, 0, :]
                    elif pooling == "mean":
                        sum_features = hidden_masked.sum(dim=1)
                        non_padding_count = expanded_mask.sum(dim=1).clamp(
                            min=1
                        )  # Clamp min to 1 to avoid division by zero
                        embedding = sum_features / non_padding_count
                    elif pooling == "concat":
                        sum_features = hidden_masked.sum(dim=1)
                        non_padding_count = expanded_mask.sum(dim=1).clamp(min=1)
                        embedding = torch.cat(
                            [
                                hidden_masked[:, 0, :],  # CLS
                                hidden_masked.max(dim=1).values,  # Max
                                sum_features / non_padding_count,  # Mean
                            ],
                            dim=1,
                        )
                    if not hidden_size:
                        hidden_size.append(embedding.shape[1])
                    yield {"embedding": embedding}

        features = Features(
            {"embedding": Array2D(shape=(len(dataset), hidden_size[0]), dtype="int32")}
        )
        result: Dataset = Dataset.from_generator(gen, features=features)
        result = result.with_format("torch").sort("_index").remove_columns("_index")
        result = concatenate_datasets(
            [result, dataset.remove_columns("_index")], axis=1
        )
        return result

    def _tokenize_fn(self, data, text_key: str, max_length: int):
        return self.tokenizer(
            data[text_key],
            truncation=True,
            padding=False,
            max_length=max_length,
        )


class SeqPreprocessor:
    """
    Class for preprocessing sequence data before converting to embedding vectors
    This
    """

    def __init__(
        self,
        seq_path: Path,
        meta: None | pl.DataFrame = None,
        anno_path: Path | str | None = None,
        id_col: str = "sample",
        split_method: SPLIT_METHODS = "bin",
        max_length: int = 512,
        include_utrs: tuple[bool, bool] = (False, False),
        utr_percent: float = 0.5,
    ):
        """Initialize preprocesser

        Parameters
        ----------
        param : argument

        Returns
        -------
        split_method : str
            Method to split sequences longer than `max_length`
            Supported methods
            - bin : Split the sequence at successive intervals equal to `max_length`
            - bakta : Split sequence at
        include_utrs : tuple[bool, bool]
            For ORF-based splitting methods, whether to include the 5' and 3' UTRs into the
                sequence, respectively
        utr_percent : float
            The percentage of intergenic region to be inlcuded as part of the UTR, measured
            from the gene start (5' UTR) or end (3' UTR)
            By default, takes half of the region from one gene, leaving the other half for the UTR of
            the downstream/upstream

        Notes
        -----

        """
        if split_method != "bin" and anno_path is None:
            raise ValueError(
                "`anno_path` file must be provided unless split_rule is `bin`!"
            )
        self.split_method: SPLIT_METHODS = split_method
        self.annotations: Path = anno_path
        self.id_col: str = id_col
        self.meta: pl.DataFrame | None = meta
        self.include_utrs: tuple[bool, bool] = include_utrs
        self.utr_percent: float = utr_percent
        accepted_suffixes = {".fasta", ".fna", ".fa"}

        self.fastas: list[Path] = [
            s for s in seq_path.iterdir() if s.suffix in accepted_suffixes
        ]

    def _sample_dict(self, record, meta: dict | None, **kwargs):
        """Boilerplate to create sequence entry for generator"""
        val = {
            "sample": id,
            "seqid": record.id,
            "sequence": str(record.seq),
            "description": record.description,
        }
        if meta is not None:
            val.update(meta)
        val.update(kwargs)
        return val

    def _process_record(
        self,
        record: SeqRecord,
        meta: dict | None,
        anno: pl.DataFrame | None = None,
    ) -> list[dict]:
        """Generate a list of sequence dicts from a single fasta record, splitting
                    the sequence according to the chosen split method

        Parameters
        ----------
        anno : pl.DataFrame
            DataFrame containing sequence annotations.
        """
        vals = []
        if self.split_method == "bin":
            acc, i = 0, 0
            split_to = len(record) - self.max_length
            while acc <= split_to:
                current = record[acc : acc + self.max_length]
                vals.append(self._sample_dict(current, meta, index=i))
                acc += self.max_length
                i += 1
        elif self.split_method == "bakta":
            filtered = anno.filter(pl.col("#Sequence Id") == record.id)
            filtered = add_intergenic(record, filtered, "Start", "Stop")
            if not filtered.is_empty():
                filtered = anno.loc[anno["#Sequence Id"] == record.id, :]
                for i, row in enumerate(filtered.iter_rows(named=True)):
                    length = row["Stop"] - row["Start"]
                    current = self._get_subsequence(
                        record, row, start="Start", stop="Stop"
                    )
                    vals.append(
                        self._sample_dict(
                            current,
                            meta,
                            strand=row["Strand"],
                            gene=row["Gene"],
                            product=row["Product"],
                            locus_tag=row["Locus Tag"],
                            type=row["Type"],
                            length=length,
                            index=i,
                        )
                    )
        return vals

    def _get_subsequence(
        self, record: SeqRecord, row: dict, start: str = "Start", stop: str = "Stop"
    ) -> SeqRecord:
        start_idx, stop_idx = row[start], row[stop]
        downstream, upstream = (
            row.get("downstream_intergenic", 0),
            row.get("upstream_intergenic", 0),
        )
        downstream = math.floor(downstream * self.utr_percent)
        upstream = math.floor(upstream * self.utr_percent)
        if not self.include_utrs[0] and not self.include_utrs[1]:
            return record[start_idx:stop_idx]
        elif self.include_utrs[0] and not self.include_utrs[1]:
            return record[start_idx + upstream : stop_idx]
        elif not self.include_utrs[0] and self.include_utrs[1]:
            return record[start_idx : stop_idx + downstream]
        else:
            return record[start_idx + upstream : stop_idx + downstream]

    def gen(self):
        """Return generator object, for use with Datasets.from_generator"""
        for fasta in self.fastas:
            id = fasta.stem
            if self.meta is not None:
                try:
                    meta = self.meta.row(
                        by_predicate=pl.col(self.id_col) == id, named=True
                    )
                except NoRowsReturnedError:
                    meta = {}
            if self.split_method == "bakta":
                anno = pl.read_csv(
                    self.annotations.joinpath(f"{id}_bakta.tsv"),
                    separator="\t",
                    skip_rows=5,
                    infer_schema_length=None,
                )
            else:
                anno = None
            for record in SeqIO.parse(fasta, "fasta"):
                for s in self._process_record(record=record, meta=meta, anno=anno):
                    yield s


@dataclass
class ModuleConfig:
    def __init__(
        self,
        record_metrics: bool = True,
        optimizer_fn: Callable | None = None,
        scheduler_fn: Callable | None = None,
        scheduler_config: dict | None = None,
        cache: str | None | Sequence = None,
        record_norm: bool = False,
        dropout_p: float = 0.2,
        init_device: str = "cpu",
        targets: tuple[Tensor] | None = None,
        outlayer_type: Literal["softmax", "regression"] = "regression",
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        task_weights : Tensor | Sequence | None
            task_weights
        log_norm : bool
            Whether to log the gradient norm
        cache : str | Sequence None
            Module attributes to cache e.g. for logging
        scaler : TorchScaler
            Fitted (ideally on entire train set) scaler that will apply transformation
            to each batch prior to training
        kwargs : Model-specific kwargs, stored in a dict
        targets : Tensor of targets that the model will observe during training
        """
        self.record_norm: bool = record_norm
        self.record: bool = record_metrics
        self._init_device: torch.device = torch.device(init_device)
        self.optimizer_fn: Callable | None = optimizer_fn
        self.scheduler_fn: Callable | None = scheduler_fn
        self.scheduler_config: dict | None = scheduler_config
        self.dropout_p: float = dropout_p
        self.cache: str | Sequence | None = cache
        self.outlayer_type: str = outlayer_type
        self.kwargs: dict = kwargs

    @property
    def init_device(self) -> torch.device:
        return self._init_device

    @init_device.setter
    def init_device(self, value: str | torch.device):
        if isinstance(value, str):
            self._init_device = torch.device(value)
        else:
            self._init_device = value


class SeqDataset:
    """
    Class representing a sequence dataset, with methods for generation
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: AutoTokenizer | str | Path,
        embedder: SeqEmbedder,
        max_length: int = 512,
    ) -> None:
        self.dataset: Dataset = load_from_disk(path)
        self.tokenizer: AutoTokenizer = tokenizer
        self.max_length: int = max_length
        self.embedder: SeqEmbedder = embedder

    def embed(self, savepath: Path | str | None) -> None:
        self.dataset = self.embedder(self.dataset)
        if savepath is not None:
            self.dataset.save_to_disk(dataset_path=savepath)

    @staticmethod
    def save_from_fastas(
        fastas: Path,
        savepath: Path,
        metadata: Path | None = None,
        mcols: Sequence | None = None,
        id_col: str = "sample",
        split_method: SPLIT_METHODS = "bakta",
        annotations: Path | str | None = None,
        max_length: int = 512,
        **kwargs,
    ) -> None:
        """Construct a dataset from fasta files and metadata,
        saving to a parquet dataset for future use

        Parameters
        ----------
        fastas : Path | str
            Directory containing fasta files. Each file is expected to be in the format
            <sample_name>.fasta|fna
        metadata : Path | str | None
            Optional path to metadata csv or tsv file describing sample attributes.
        mcols : Sequence | None
            If None, all metadata columns will be included in the dataset object. Otherwise,
            only these specific columns
        """
        if metadata:
            metadata = metadata if isinstance(metadata, Path) else Path(metadata)
            sep = "\t" if metadata.suffix == ".tsv" else ","
            meta: pl.DataFrame | None = pl.read_csv(
                metadata, separator=sep, infer_schema_length=None
            ).unique(id_col)
            if mcols is not None:
                meta = meta.select(mcols)
        else:
            meta = None
        spp = SeqPreprocessor(
            seq_path=fastas,
            meta=meta,
            id_col=id_col,
            split_method=split_method,
            anno_path=annotations,
            max_length=max_length,
            **kwargs,
        )
        dataset = Dataset.from_generator(spp.gen)
        dataset.save_to_disk(dataset_path=savepath)


# TODO:
# 1. Find a way to format fasta files into a huggingface dataset, along with all their
# annotation data. Probably use generator style DONE
# 2. Set up a function to tokenize the dataset
# 3. Figure out what's up with the datacollator
# 4. Extract the hidden state from seqLens output
