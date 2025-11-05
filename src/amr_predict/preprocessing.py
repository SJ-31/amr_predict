#!/usr/bin/env ipython
from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path
from subprocess import run
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Callable, Iterator, Literal, TypeAlias

import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from amr_predict.utils import add_intergenic, join_within, read_tabular, split_features
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
EMBEDDING_METHODS: TypeAlias = Literal["seqLens", "Evo2", "kmer", "feature_presence"]


class SeqEmbedder:
    def __init__(
        self,
        method: EMBEDDING_METHODS = "seqLens",
        **kwargs,
    ):
        self.method: EMBEDDING_METHODS = method
        self.kwargs: dict = kwargs

    def __call__(self, dataset: Dataset | None = None) -> Dataset:
        if self.method not in {"kmer", "feature_presence"} and dataset is None:
            raise ValueError("The selected method requires a preprocessed dataset")
        if self.method == "seqLens":
            return self._seqlens_embed(dataset, **self.kwargs)
        elif self.method == "Evo2":
            raise ValueError("Not implemented yet")
        elif self.method == "kmer":
            return self._kmer_embed(**self.kwargs)
        elif self.method == "feature_presence":
            raise ValueError("Not implemented yet")
        else:
            raise ValueError("Given method not supported!")

    def _kmer_embed(
        self,
        fastas: Path,
        id_col: str = "sample",
        metadata: Path | None = None,
        k: int = 5,
        key: str = "x",
        accepted_suffixes=(".fasta", ".fna", ".fa"),
    ) -> Dataset:
        """Process sample-level fasta files so that each sample is represented
        by a feature vector of kmer counts
        """

        def count_kmers(fasta: Path) -> pl.DataFrame:
            # Count kmers quickly with kmc
            with TemporaryDirectory() as dir:
                out = Path(dir).joinpath("result.tsv")
                run(f"kmc -fm -k{k} {fasta} tmp .", shell=True)
                run(f"kmc_dump tmp {out}", shell=True)
                counts = (
                    pl.read_csv(
                        out,
                        separator="\t",
                        new_columns=["kmer", "count"],
                        has_header=False,
                    )
                    .with_columns(pl.lit(fasta.stem).alias(id_col))
                    .pivot("kmer", values="count")
                )
            return counts

        dfs = [
            count_kmers(f) for f in fastas.iterdir() if f.suffix in accepted_suffixes
        ]
        df = pl.concat(dfs, how="diagonal_relaxed").fill_null(0)
        kmer_cols = df.drop("sample").columns
        if metadata is not None:
            meta = read_tabular(metadata)
            df = df.join(meta, on=id_col)
        arr = np.array(df.select(kmer_cols))
        variance = arr.var(axis=0)
        arr = arr[:, variance != 0]
        dct = df.drop(kmer_cols).to_dict()
        dct[key] = arr
        return Dataset.from_dict(dct).with_format("torch")

    def _evo2_embed(
        self,
        dset: Dataset,
        text_key: str = "sequence",
    ) -> Dataset:
        dset = dset.add_column("uid", list(range(len(dset)))).sort("uid")
        df: pl.DataFrame = dset.to_polars().with_columns(
            (
                ">"
                + pl.concat_str(
                    [pl.col("uid"), pl.col(text_key)], separator="\n"
                ).alias("fasta")
            )
        )
        seqs = df["seqid"].unique()

        def gen():
            for seq in seqs:
                for embedding in self._evo2_embed_one(df, seq):
                    yield embedding

        result: Dataset = Dataset.from_generator(gen)
        result = result.with_format("torch").sort("uid").remove_columns("uid")
        dset = dset.remove(text_key)
        result = concatenate_datasets([result, dset], axis=1)
        return result

    def _evo2_embed_one(self, df: pl.DataFrame, cur_seq: str) -> Iterator[dict]:
        df = df.filter(pl.col("seqid") == cur_seq)
        result = []
        with NamedTemporaryFile("w") as f:
            f.write("\n".join(df["fasta"]))
            # evo2 -i f.name -o
            embedding_file = ""
        # for i, g in embedding_file:

        # TODO: run evo2 to convert fasta into embeddings
        result = None
        return result

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
    """

    def __init__(
        self,
        seq_path: Path,
        anno_path: Path | str | None = None,
        split_method: SPLIT_METHODS = "bin",
        max_length: int = 512,
        include_utrs: tuple[bool, bool] = (False, False),
        utr_amount: tuple[float | int, float | int] | None = None,
        upstream_context: int = 200,
        downstream_context: int = 200,
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
        utr_amount : tuple
            The percentage or number of bases of intergenic region to be
            included as part of the UTR, measured
            from the gene start (5' UTR) or end (3' UTR)
            By default, takes half of the region from one gene, leaving the other half for the UTR of
            the downstream/upstream
        upstream_context : int
            Number of upstream bases to include into the embedding, relevant to all
            embedded sequences e.g. chunks of genes
            This is not included if `include_utrs` and the sequence to embed is at the
            feature 5' end
        downstream_context : int
            Number of downstream bases to include into the embedding, relevant to all
            embedded sequences e.g. chunks of genes
            This is not included if `include_utrs` and the sequence to embed is at the
            feature 3' end

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
        self.utr_amount: tuple[float | int, float | int] | None = utr_amount
        self.upstream_context: int = upstream_context
        self.downstream_context: int = downstream_context
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
                filtered = split_features(
                    filtered,
                    self.max_length,
                    "Start",
                    "Stop",
                    indicate_ends=True,
                    prefix="chunk",
                )
                filtered = add_intergenic(record, filtered, "Start", "Stop")
                for i, row in enumerate(filtered.iter_rows(named=True)):
                    length = row["Stop"] - row["Start"]
                    current, indices = self._get_subsequence(
                        record, row, start="chunk_Start", stop="chunk_Stop"
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
                        is_5prime=row.get("is_5prime"),
                        is_3prime=row.get("is_3prime"),
                        seqindex=i,
                        # Start, stop indices after (possible) context widening
                        start=indices[0],
                        stop=indices[1],
                        # Original start, stop indices of features
                        old_start=row["Start"],
                        old_stop=row["Stop"],
                    )
                    vals.append(val)
        return vals

    def _get_subsequence(
        self, record: SeqRecord, row: dict, start: str = "Start", stop: str = "Stop"
    ) -> tuple[SeqRecord, tuple[int, int]]:
        start_idx, stop_idx = row[start], row[stop]
        # Get widths of actual downstream, upstream sequences for percentage-based
        # widening. Overriden if widening dowstream, upstream by flat number of bases
        downstream, upstream = (
            row.get("downstream_intergenic", 0),
            row.get("upstream_intergenic", 0),
        )
        if self.utr_amount is not None:
            if not row.get("is_5prime", True):
                upstream = 0
            elif isinstance(self.utr_amount[0], float):
                upstream = max(0, math.floor(upstream * self.utr_amount[0]))
            else:
                upstream = max(0, self.utr_amount[0])
            if not row.get("is_3prime", True):
                downstream = 0
            if isinstance(self.utr_amount[1], float):
                downstream = max(0, math.floor(downstream * self.utr_amount[1]))
            else:
                downstream = max(0, self.utr_amount[1])

        if upstream == 0 and self.upstream_context:
            upstream += self.upstream_context
        if downstream == 0 and self.downstream_context:
            downstream += self.downstream_context

        if self.include_utrs[0] and not self.include_utrs[1]:
            start_idx = max(0, start_idx - upstream)
        elif not self.include_utrs[0] and self.include_utrs[1]:
            stop_idx = min(stop_idx + downstream, len(record))
        elif self.include_utrs[0] and self.include_utrs[1]:
            start_idx = max(0, start_idx - upstream)
            stop_idx = min(stop_idx + downstream, len(record))
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
