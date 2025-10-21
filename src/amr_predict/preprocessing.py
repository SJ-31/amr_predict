#!/usr/bin/env ipython
from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeAlias

import polars as pl
import torch
from amr_predict.utils import add_intergenic, join_within, read_tabular
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from datasets import concatenate_datasets
from datasets.arrow_dataset import Dataset
from datasets.load import load_from_disk
from polars.exceptions import NoDataError
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding
from transformers.modeling_outputs import MaskedLMOutput

SPLIT_METHODS: TypeAlias = Literal["bin", "bakta"]
EMBEDDING_METHODS: TypeAlias = Literal["seqLens", "Evo2"]


class SeqEmbedder:
    def __init__(
        self,
        method: EMBEDDING_METHODS = "seqLens",
        **kwargs,
    ):
        self.method: EMBEDDING_METHODS = method
        self.kwargs: dict = kwargs

    def __call__(self, dataset: Dataset) -> Dataset:
        if self.method == "seqLens":
            return self._seqlens_embed(dataset, **self.kwargs)
        elif self.method == "Evo2":
            raise ValueError("Not implemented yet")
        else:
            raise ValueError("Given method not supported!")

    def _seqlens_embed(
        self,
        dset: Dataset,
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
            Column name of `dset` containing the sequences
        """
        torch.set_default_dtype(torch.float32)
        os.environ["HF_HOME"] = huggingface
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_key)
        model = AutoModelForMaskedLM.from_pretrained(model_key)
        model.to(device)
        model.config.output_hidden_states = True
        to_keep = {"uid", "label"}
        dset = dset.add_column("uid", list(range(len(dset))))
        to_remove = [c for c in dset.column_names if c not in to_keep]
        tokenized = dset.map(
            lambda x: self._tokenize(
                x, tokenizer=tokenizer, text_key=text_key, max_length=512
            ),
            batched=True,
            remove_columns=to_remove,
        )
        dset = dset.remove_columns(text_key)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # TODO: is there any reason to change this from the default?
        loader = DataLoader(tokenized, batch_size=batch_size, collate_fn=data_collator)

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
                    for e, uid in zip(
                        torch.unbind(embedding, axis=0), torch.unbind(batch["uid"])
                    ):
                        yield {"embedding": e, "uid": uid}

        result: Dataset = Dataset.from_generator(gen)
        result = result.with_format("torch").sort("uid").remove_columns("uid")
        result = concatenate_datasets([result, dset.sort("uid")], axis=1)
        return result

    @staticmethod
    def _tokenize(data, tokenizer, text_key: str, max_length: int):
        return tokenizer(
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
        anno_path: Path | str | None = None,
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
        self.include_utrs: tuple[bool, bool] = include_utrs
        self.max_length: int = max_length
        self.utr_percent: float = utr_percent
        accepted_suffixes = {".fasta", ".fna", ".fa"}

        self.fastas: list[Path] = [
            s for s in seq_path.iterdir() if s.suffix in accepted_suffixes
        ]

    def _sample_dict(self, sample, record, **kwargs) -> dict:
        """Boilerplate to create sequence entry for generator"""
        val = {
            "sample": sample,
            "seqid": record.id,
            "sequence": str(record.seq),
            "description": record.description,
        }
        val.update(kwargs)
        return val

    def _process_record(
        self,
        sample: str,
        record: SeqRecord,
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
                val: dict = self._sample_dict(
                    sample=sample,
                    record=current,
                    seqindex=i,
                    start=acc,
                    stop=acc + self.max_length,
                )
                vals.append(val)
                acc += self.max_length
                i += 1
        elif self.split_method == "bakta":
            filtered = anno.filter(pl.col("#Sequence Id") == record.id)
            if not filtered.is_empty():
                filtered = add_intergenic(record, filtered, "Start", "Stop")
                for i, row in enumerate(filtered.iter_rows(named=True)):
                    length = row["Stop"] - row["Start"]
                    current, indices = self._get_subsequence(
                        record, row, start="Start", stop="Stop"
                    )
                    val = self._sample_dict(
                        sample=sample,
                        record=current,
                        strand=row["Strand"],
                        gene=row["Gene"],
                        product=row["Product"],
                        locus_tag=row["Locus Tag"],
                        type=row["Type"],
                        length=length,
                        seqindex=i,
                        start=indices[0],
                        stop=indices[1],
                    )
                    vals.append(val)
        return vals

    def _get_subsequence(
        self, record: SeqRecord, row: dict, start: str = "Start", stop: str = "Stop"
    ) -> tuple[SeqRecord, tuple[int, int]]:
        start_idx, stop_idx = row[start], row[stop]
        downstream, upstream = (
            row.get("downstream_intergenic", 0),
            row.get("upstream_intergenic", 0),
        )
        downstream = max(0, math.floor(downstream * self.utr_percent))
        upstream = max(0, math.floor(upstream * self.utr_percent))
        if self.include_utrs[0] and not self.include_utrs[1]:
            start_idx += upstream
        elif not self.include_utrs[0] and self.include_utrs[1]:
            stop_idx += downstream
        elif self.include_utrs[0] and self.include_utrs[1]:
            start_idx += upstream
            stop_idx += downstream
        return record[start_idx:stop_idx], (start_idx, stop_idx)

    def gen(self):
        """Return generator object, for use with Datasets.from_generator"""
        for fasta in self.fastas:
            id = fasta.stem
            if self.split_method == "bakta":
                try:
                    anno = pl.read_csv(
                        self.annotations.joinpath(f"{id}_bakta.tsv"),
                        separator="\t",
                        skip_rows=5,
                        infer_schema_length=None,
                    )
                except FileNotFoundError:
                    print(
                        f"WARNING: bakta file {id}_bakta.tsv not found! Skipping this sample"
                    )
                    anno = None
            else:
                anno = None
            if anno is not None or self.split_method != "bakta":
                for record in SeqIO.parse(fasta, "fasta"):
                    for s in self._process_record(sample=id, record=record, anno=anno):
                        yield s


class SeqDataset:
    """
    Class representing a sequence dataset, with methods for generation
    """

    def __init__(
        self,
        path: str | Path,
        embedder: SeqEmbedder,
        max_length: int = 512,
    ) -> None:
        self.dataset: Dataset = load_from_disk(path)
        self.max_length: int = max_length
        self.embedder: SeqEmbedder = embedder

    def embed(self, savepath: Path | str | None) -> None:
        self.dataset = self.embedder(self.dataset)
        if savepath is not None:
            self.dataset.save_to_disk(dataset_path=savepath)

    @staticmethod
    def save_from_fastas(
        fastas: Path | str,
        savepath: Path,
        metadata: Path | None = None,
        mcols: Sequence | None = None,
        id_col: str = "sample",
        split_method: SPLIT_METHODS = "bakta",
        annotations: Path | str | None = None,
        seq_metadata: Path | str | None = None,
        seq_id_col: str = "seqid",
        seq_start: str = "start",
        seq_stop: str = "stop",
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
            Optional path to csv or tsv file describing sample attributes.
            Each row should uniquely identify a sample
        seq_metadata : Path | str | None
            Optional path to csv or tsv file containing sequence-level attributes for each sample
            e.g. amr gene annotations
            Unique entries should be defined by `id_col` and `seqid_col`
            Content is similar to individual files contained in `annotations`, but aggregated
            for the cohort
        mcols : Sequence | None
            If None, all metadata columns will be included in the dataset object. Otherwise,
            only these specific columns
        """

        spp = SeqPreprocessor(
            seq_path=Path(fastas) if isinstance(fastas, str) else fastas,
            split_method=split_method,
            anno_path=annotations,
            max_length=max_length,
            **kwargs,
        )
        dataset = Dataset.from_generator(spp.gen)
        if metadata:
            meta = read_tabular(metadata).rename({id_col: "sample"}).unique("sample")
            if mcols is not None:
                meta = meta.select(mcols)
            to_combine = Dataset.from_polars(
                dataset.select_columns("sample")
                .to_polars()
                .join(meta, how="left", on="sample", maintain_order="left")
                .drop("sample")
            )
            dataset = concatenate_datasets([dataset, to_combine], axis=1)
        if seq_metadata:
            try:
                required_cols = ["sample", "seqid", "start", "stop"]
                seq_meta: pl.DataFrame = (
                    read_tabular(seq_metadata)
                    .rename({seq_id_col: "seqid", seq_start: "start", seq_stop: "stop"})
                    .unique(required_cols)
                )
                entries_within: pl.DataFrame = join_within(
                    dataset.select_columns(required_cols).to_polars(),
                    seq_meta,
                    initial_join=["sample", "seqid"],
                    start_col="start",
                    stop_col="stop",
                )
                exclude = set(dataset.column_names)
                to_drop = [c for c in entries_within.columns if c in exclude]
                entries_within = Dataset.from_polars(entries_within.drop(to_drop))
                dataset = concatenate_datasets([dataset, entries_within], axis=1)
            except NoDataError:
                print("No sequence metadata provided")
        dataset.save_to_disk(dataset_path=savepath)
