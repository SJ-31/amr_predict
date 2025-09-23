#!/usr/bin/env ipython
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import datasets as hd
import pandas as pd
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from datasets.arrow_dataset import Dataset
from torch import Tensor

# from torch.utils.data import Dataset
from transformers import AutoTokenizer, pipeline

SPLIT_METHODS: TypeAlias = Literal["bin", "bakta"]


class SeqPreprocessor:
    """
    Class for preprocessing sequence data before converting to embedding vectors
    This
    """

    def __init__(
        self,
        seq_path: Path,
        meta_path: Path | str | None = None,
        anno_path: Path | str | None = None,
        id_col: str = "sample",
        split_rule: SPLIT_METHODS = "bin",
        max_length: int = 512,
    ):
        if split_rule != "bin" and anno_path is None:
            raise ValueError(
                "`anno_path` file must be provided unless split_rule is `bin`!"
            )
        self.split_method: SPLIT_METHODS = split_rule
        self.annotations: Path = anno_path
        self.id_col: str = id_col
        if meta_path:
            meta_path = meta_path if isinstance(meta_path, Path) else Path(meta_path)
            sep = "\t" if meta_path.suffix == ".tsv" else ","
            self.meta: pd.DataFrame | None = pd.read_csv(
                meta_path, sep=sep
            ).drop_duplicates(id_col)
            self.meta.index = self.meta[id_col]
            self.meta = self.meta.drop(id_col, axis=1)
        else:
            self.meta = None
        accepted_suffixes = {".fasta", ".fna", ".fa"}

        self.fastas: list[Path] = [
            s for s in seq_path.iterdir() if s.suffix in accepted_suffixes
        ]

    def _sample_dict(self, record, meta: dict | None, **kwargs):
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
        anno: pd.DataFrame | None = None,
    ) -> list[dict]:
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
            filtered = anno.loc[anno["#Sequence Id"] == record.id, :]
            for i, row in filtered.iterrows():
                current = record[row["Start"] : row["Stop"]]
                vals.append(
                    self._sample_dict(
                        current,
                        meta,
                        Strand=row["Strand"],
                        Gene=row["Gene"],
                        Product=row["Product"],
                        Locus_tag=row["Locus Tag"],
                        Type=row["Type"],
                        index=i,
                    )
                )
        return vals

    def gen(self):
        for fasta in self.fastas:
            id = fasta.stem
            if self.meta is not None:
                try:
                    meta = self.meta.loc[id].to_dict()
                except KeyError:
                    meta = {}
            if self.split_method == "bakta":
                anno = pd.read_csv(
                    self.annotations.joinpath(f"{id}_bakta.tsv"), sep="\t", skiprows=5
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

    Parameters
    ----------
    tokenizer :

    Returns
    -------


    Notes
    -----

    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: AutoTokenizer | str | Path,
        max_length: int = 512,
    ) -> None:
        self.tokenizer: AutoTokenizer = tokenizer
        self.max_length: int = max_length
        pass

    @staticmethod
    def save_from_fastas(
        fastas: Path,
        savepath: Path,
        metadata: Path | None = None,
        mcols: Sequence | None = None,
        id_col: str = "sample",
        split_rule: str = "ORF",
        annotations: Path | str | None = None,
        max_length: int = 512,
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
        split_rule : str
            Method to split sequences longer than `max_length`
            Supported methods
            - bin : Split the sequence at successive intervals equal to `max_length`

        Returns
        -------


        Notes
        -----

        """
        ...
        # dataset.save_to_disk(dataset_path=savepath)

    def _tokenize_fn(self, data):
        return self.tokenizer(
            data[self.text_key],
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )

    def _tokenize(self, dataset: Dataset):
        dataset.map(self._tokenize_fn, batched=True, remove_columns=[])
        return self.tokenizer()


# TODO:
# 1. Find a way to format fasta files into a huggingface dataset, along with all their
# annotation data. Probably use generator style
# 2. Set up a function to tokenize the dataset
# 3. Figure out what's up with the datacollator
# 4. Extract the hidden state from seqLens output


# class SeqDataset(Dataset):
#     """Torch dataset with sequences as samples

#     Parameters
#     ----------
#     metadata : DataFrame
#         df containing sample information
#     vars : tuple[str]
#         columns of `metadata` to include in the Dataset object, in order
#     tokenizer : Path | AutoTokenizer
#         Tokenizer to process the sequence data or the path to a pretrained one
#     dir : Path
#         Directory containing sequence files in fasta format. The files must be named
#         <id>.fasta or <id>.fa where <id> are elements of metadata['id_col']
#     """

#     @staticmethod  # TODO: best way of saving torch dataset to disk?
#     def load() -> SeqDataset: ...

#     def __init__(
#         self,
#         dir: Path,
#         metadata: pd.DataFrame,
#         vars: tuple[str],
#         tokenizer: Preprocessor,
#         id_col: str = "acc",
#     ) -> None:
#         super().__init__()
#         self.saved_path: Path | None = None
#         self.meta: pd.DataFrame = metadata
#         self.X: Tensor

#     def save(self, path: Path): ...

#     def tokenize(self): ...

#     @property
#     def shape(self) -> tuple:
#         return self.X.shape

#     def __len__(self) -> int:
#         return self.X.shape[0]

#     # Might want to just tokenize at init
