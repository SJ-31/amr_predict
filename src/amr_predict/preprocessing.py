#!/usr/bin/env ipython
from __future__ import annotations

import json
import math
import os
import uuid
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
from typing import Callable, Literal, TypeAlias, get_args

import numpy as np
import pandera.polars as pa
import polars as pl
import polars.selectors as cs
import skbio
import torch
from amr_predict.enums import BasicPoolings
from amr_predict.pooling import pool_tensor
from amr_predict.utils import (
    EmbeddingCache,
    add_intergenic,
    join_within,
    read_tabular,
    split_features,
    translate_df,
)
from datasets import concatenate_datasets, disable_progress_bar
from datasets.arrow_dataset import Dataset
from datasets.load import load_from_disk
from loguru import logger
from polars.exceptions import ComputeError, NoDataError
from skbio import DNA
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding
from transformers.modeling_outputs import MaskedLMOutput

SPLIT_METHODS: TypeAlias = Literal["bin", "bakta"]
EMBEDDING_METHODS: TypeAlias = Literal[
    "seqLens", "Evo2", "kmer", "feature_presence", "esm"
]
ESM_MODELS: TypeAlias = Literal[
    "esm2_t6_8m_UR50D",
    "esm2_t33_650m_UR50D",
    "esm3-open",
    "esmc_600m",
    "esmc_300m",
    "esmc_600m_synthyra",
    "esmc_300m_synthyra",
]

disable_progress_bar()
logger.disable("amr_predict")


class SeqEmbedder:
    def __init__(
        self,
        method: EMBEDDING_METHODS = "seqLens",
        workdir: Path | None = None,
        save_mode: Literal["tokens", "seqs", "both"] = "seqs",
        save_interval: int = 10,
        only_cache: bool = True,
        token_prop: float | None = None,
        **kwargs,
    ):
        self.method: EMBEDDING_METHODS = method
        self.kwargs: dict = kwargs
        self.token_prop: float | None = token_prop
        self.save_mode: Literal["tokens", "seqs", "both"] = save_mode
        if self.save_mode != "seqs":
            logger.warning(
                "Embedder set to save tokens. Ensure you have enough memory and space"
            )
        self.only_cache: bool = only_cache
        self.save_interval = save_interval
        if workdir is None and self.method not in {"kmer", "feature_presence"}:
            name = f"{method}-work_{uuid.uuid4().hex}"
            self.workdir = Path().cwd().joinpath(name)
            self.workdir.mkdir()
        else:
            self.workdir = workdir

    @property
    def with_tokens(self) -> bool:
        return self.save_mode != "seqs"

    def __call__(self, dataset: Dataset | None) -> Dataset | None:
        if self.method not in {"kmer", "feature_presence"} and dataset is None:
            raise ValueError("The selected method requires a preprocessed dataset")
        if self.method == "seqLens":
            return self._seqlens_embed(dset=dataset, **self.kwargs)
        elif self.method == "Evo2":
            return self._evo2_embed(dset=dataset, **self.kwargs)
        elif self.method == "esm":
            return self._esm_embed(dset=dataset, **self.kwargs)
        elif self.method == "kmer":
            return self._kmer_embed(**self.kwargs)
        elif self.method == "feature_presence":
            return self._feature_presence_embed(**self.kwargs)
        else:
            raise ValueError("Given method not supported!")

    def _embed_from_files(
        self,
        id_col,
        path: Path,
        metadata: Path | None,
        accepted_suffixes: tuple,
        key: str,
        embed_fn: Callable,
        id_rename: str | None = None,
        pattern: str | None = None,
        features: tuple | list | None = None,
        save_features_to: Path | None = None,
    ) -> Dataset:
        """Helper function to generate embeddings from files

        Parameters
        ----------
        path : Path
            Directory or file.
            In the former, each file represents a single sample, and
            `embed_fn` is called on each file
            In the latter, a file aggregating multiple samples
        """
        if path.is_dir():
            to_iter = path.glob(pattern) if pattern else path.iterdir()
            dfs = [embed_fn(f) for f in to_iter if f.suffix in accepted_suffixes]
            df: pl.DataFrame = pl.concat(dfs, how="diagonal_relaxed").fill_null(0)
        else:
            df = embed_fn(path).fill_null(0)
        to_drop = df.drop(id_col).columns
        feature_cols = features or to_drop
        logger.info(f"{len(feature_cols)} features")
        if metadata is not None:
            meta = read_tabular(metadata)
            df = df.join(meta, on=id_col)
        if id_rename is not None:
            df = df.rename({id_col: id_rename})
        arr = np.array(df.select(feature_cols))
        # Drop all features from the dataset, leaving them only as columns in the sample
        # array
        if not features:
            variance: np.ndarray = arr.var(axis=0)
            feature_mask = variance != 0
            n_removed = (~feature_mask).sum()
            total = len(feature_mask)
            n_left = total - n_removed
            logger.info(
                f"Removing {n_removed} features with 0 variance. \n{n_left} remaining."
            )
            kept_indices = np.where(feature_mask)[0]
            logger.info("kept {}", kept_indices)
            kept = np.array(feature_cols)[feature_mask].tolist()
            assert len(kept) == n_left
            arr = arr[:, feature_mask]
            if arr.shape[1] == 0:
                raise ValueError("No features remaining after variance filtering")
            if save_features_to is not None:
                save_features_to.write_text("\n".join(kept))
        else:
            kept = [o for o in to_drop if o in features]
            logger.info(
                "Dropping {} features not found in passed `feature` parameter",
                len(to_drop) - len(kept),
            )
        dct = df.drop(to_drop).to_dict()
        dct[key] = arr
        return Dataset.from_dict(dct).with_format("torch")

    def _kmer_embed(
        self,
        fastas: Path,
        id_col: str = "sample",
        metadata: Path | None = None,
        k: int = 5,
        key: str = "x",
        max_kmer_count: int = 255,
        save_stats: Path | None = None,
        **kws,
    ) -> Dataset:
        """Process sample-level fasta files so that each sample is represented
        by a feature vector of kmer counts
        """
        tmp_stats = defaultdict(list)

        def count_kmers(fasta: Path) -> pl.DataFrame:
            # Count kmers quickly with kmc
            with TemporaryDirectory() as dir:
                out = Path(dir) / "result.tsv"
                stat_file = Path(dir) / "stats.json"
                kmc_proc = run(
                    f"kmc -fm -k{k} -j{stat_file} -cs{max_kmer_count} {fasta} tmp .",
                    shell=True,
                    capture_output=True,
                )
                kmc_proc.check_returncode()
                run(f"kmc_dump tmp {out}", shell=True)
                counts = (
                    pl.read_csv(
                        out,
                        separator="\t",
                        new_columns=["kmer", "count"],
                        has_header=False,
                    )
                    .with_columns(pl.lit(fasta.stem).alias(id_col))
                    .pivot("kmer", values="count", aggregate_function="sum")
                )
                with open(stat_file, "r") as st:
                    stats = json.load(st)
                for key, v in stats.items():
                    if key == "Stats":
                        for skey, val in stats[key].items():
                            tmp_stats[skey].append(val)
                    else:
                        tmp_stats[key].append(v)
                tmp_stats["file"].append(fasta.name)
            return counts

        try:
            embedded = self._embed_from_files(
                id_col=id_col,
                path=fastas,
                metadata=metadata,
                accepted_suffixes=(".fasta", ".fna", ".fa"),
                key=key,
                embed_fn=count_kmers,
                **kws,
            )
        except ValueError as e:
            kmer_stats = pl.DataFrame(tmp_stats)
            logger.debug("stats {}", kmer_stats)
            if (
                e.args == "No features remaining after variance filtering"
                and len(kmer_stats["#Total no. of k-mers"].unique()) > 1
            ):
                raise ValueError(
                    """
                    No features remaining after variance filtering, but the k-mer counts are unique.
                    Try increasing k or the maximum k-mer count
                    """
                )
            raise e
        kmer_stats = pl.DataFrame(tmp_stats)
        if save_stats is not None:
            kmer_stats.write_csv(save_stats)
        return embedded

    def _feature_presence_embed(
        self,
        fasta_annotations: Path,
        feature_cols: str | list[str],
        feature_whitelist: tuple | dict | str | Path = (),
        feature_blacklist: tuple | dict | str | Path = (),
        recode_file: Path | str | None = None,
        recoding_join_cols: Sequence | None = None,
        id_col: str = "sample",
        id_regexp: str = "",
        metadata: Path | None = None,
        key: str = "x",
        read_kws: dict | None = None,
        metadata_pattern: str | None = None,
        **kws,
    ):
        """Encode sample as a binary vector describing the presence of specific sequence
        features.

        Parameters
        ----------
        fasta_annotations : Path
            A path to tabular files annotating the fasta samples.
            The following conventions are required:
                - Each entry (line) is an annotation to a sequence feature
                - If a single file and not a directory, there must be a column denoting
                    sample identity
                - If a directory, the file stem must be the sample name
        feature_cols : str | list[str]
            Column(s) in the files naming the feature in the annotation entry
        feature_blacklist : tuple | dict | str | Path
            Tuple of named features to exclude
            Can also be a dictionary mapping the name of a feature column to a tuple of feeatures to exclude
        feature_whitelist : tuple | dict | str | Path
            Named features to include. Effect is mutually exclusive to `feature_blacklist`
        read_kws : dict
            keyword arguments passed to `pl.read_csv`
        id_regexp : str
            A regex pattern containing a single capture group to extract sample names from
            `id_col`. Only used when `fasta_annotations` is a single file
        recode_file : Path | str | None
            A tabular file defining how to map from old feature labels to new ones. Requires
            a single column "new" that contains the new features. All other columns are
            used to join the dataset on `recoding_join_cols`
        recoding_join_cols : Sequence | None
            Columns in the dataset with which to join the recodings by
        """
        read_kws = read_kws or {}
        is_annotations_dir: bool = fasta_annotations.is_dir()
        feature_whitelist = feature_whitelist or kws.pop("features", ())

        if (recode_file and not recoding_join_cols) or (
            recoding_join_cols and not recode_file
        ):
            raise ValueError(
                "Both `recode_file` and `recoding_join_cols` must be given"
            )
        if recode_file:
            recode: pl.DataFrame | None = read_tabular(recode_file)
            join_cols = [c for c in recode.columns if c != "new"]
            schema_dict = {c: pa.Column(nullable=True) for c in join_cols}
            schema_dict.update({"new": pa.Column("string", unique=True)})
            schema: pa.DataFrameSchema = pa.DataFrameSchema(
                schema_dict, unique=join_cols + ["new"]
            )
            schema.validate(recode)
        else:
            recode = None
            join_cols = None

        def helper(file: Path, fcols: str | list[str]) -> pl.DataFrame:
            try:
                df: pl.DataFrame = read_tabular(file, **read_kws)
            except ComputeError as e:
                logger.exception(f"Failed to read file {file}")
                raise e

            if recode is not None:
                df = df.join(recode, left_on=recoding_join_cols, right_on=join_cols)
                fcols = ["new"]

            for fspec, is_blacklist in zip(
                (feature_whitelist, feature_blacklist), (False, True)
            ):
                if isinstance(fspec, str) or isinstance(fspec, Path):
                    with open(fspec, "r") as f:
                        fspec = f.read().splitlines()
                if fspec and isinstance(fspec, dict):
                    expr = [
                        pl.col(k).is_in(v).alias(f"_has_{k}") for k, v in fspec.items()
                    ]
                    if is_blacklist:
                        expr = [e.not_() for e in expr]
                    df = (
                        df.with_columns(*expr)
                        .filter(pl.any_horizontal(cs.starts_with("_has")))
                        .drop(cs.starts_with("_has"))
                    )
                elif fspec:
                    df = df.filter(pl.any_horizontal(pl.col(fcols).is_in(fspec)))

            if is_annotations_dir:
                df = df.select(fcols).with_columns(pl.lit(file.stem).alias(id_col))

            else:
                to_select = (
                    [id_col] + fcols if isinstance(fcols, list) else [id_col, fcols]
                )
                df = df.select(to_select)
            if id_regexp:
                df = df.with_columns(
                    pl.col(id_col).str.extract(id_regexp, group_index=1)
                )

            df = (
                df.unpivot(index=id_col)
                .filter(pl.col("value").is_not_null())
                .with_columns(pl.lit(1).alias("tmp"))
                .drop("variable")
                .unique("value")
                .pivot("value", values="tmp")
            )
            return df

        return self._embed_from_files(
            id_col=id_col,
            path=fasta_annotations,
            metadata=metadata,
            accepted_suffixes=(".tsv", ".csv"),
            key=key,
            embed_fn=lambda x: helper(x, feature_cols),
            pattern=metadata_pattern,
            **kws,
        )

    def _esm_embed(
        self,
        dset: Dataset,
        huggingface: str,
        text_key: str = "sequence",
        model: ESM_MODELS = "esmc_600m",
        pooling: Literal["cls"] = "mean",
        hidden_layer: int | None = None,
        batch_size=5,
        degenerate_handling: Literal["ignore", "random", "error"] = "random",
        save_dset: Path | None = None,
        from_nucleotide: bool = True,
        **kws,
    ) -> Dataset | None:
        """Embed nucleotide sequences with an esm model after translating into protein

        Parameters
        ----------
        degenerate_handling : str
            How to handle degenerate bases in the nucleotide sequence during translation
            - random: expand all possible degenerates and take the first option.
                This is random because there is no guaranteed order to skbio.DNA.expand_degenerates
            - ignore: do not embed sequence
            - error: raise an error
            For "random", degenerate sequences are indicated in the new "dna_degenerate" column

        pooling : str
            How to pool per-residue embeddings

        hidden_layer : int | None
            If not None, the index of the hidden layer of the esm model to take as an embedding
            Otherwise, the key "embedding" is taken
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tkey = f"{text_key}_aa"
        df: pl.DataFrame = dset.to_polars()
        if tkey not in df.columns and from_nucleotide:
            df = translate_df(
                df, text_key, new_col=None, degenerate_handling=degenerate_handling
            )
            if save_dset is not None:
                logger.info("Saving dataset with translated aa column")
                Dataset.from_polars(df).save_to_disk(save_dset)
        else:
            tkey = text_key

        torch.set_default_dtype(torch.float32)
        os.environ["HF_HOME"] = huggingface
        get_hidden: bool = hidden_layer is not None
        cache: EmbeddingCache = EmbeddingCache(
            self.workdir,
            save_mode=self.save_mode,
            save_interval=self.save_interval,
            token_prop=self.token_prop,
        )
        if model.startswith("esm2"):
            os.environ["HF_HOME"] = huggingface
            from interplm.esm.embed import embed_single_sequence

            def esm2_interplm(proteins):
                layer = hidden_layer or 4
                for prot in proteins:
                    token_embeddings = embed_single_sequence(
                        prot, model_name=model, layer=layer
                    )
                    tokens = token_embeddings if self.with_tokens else None
                    pooled = pool_tensor(token_embeddings, pooling, **kws)
                    yield prot, pooled, tokens

            cache.save(df[tkey], fn=esm2_interplm, batch_size=batch_size)
        elif not model.endswith("synthyra"):
            from esm.sdk.api import ESMProtein, LogitsConfig

            lconf = LogitsConfig(
                return_embeddings=not get_hidden, return_hidden_states=get_hidden
            )
            if model == "esm3-open":
                from esm.models.esm3 import ESM3

                client = ESM3.from_pretrained(model, device=device)
            else:
                from esm.models.esmc import ESMC

                client = ESMC.from_pretrained(model)
            to_get = "hidden_states" if hidden_layer is not None else "embeddings"

            def esm_official(proteins):
                for prot in proteins:
                    encoded = client.encode(ESMProtein(sequence=prot))
                    logits = client.logits(encoded, lconf)
                    target: Tensor = getattr(logits, to_get)
                    # If `get_hidden`, target has shape
                    #    (n_hidden, 1, sequence_len, dim_model)
                    # Otherwise
                    #  (1, sequence_len, dim_model)
                    if get_hidden and self.with_tokens:
                        tokens = target[hidden_layer, 0, 1:-1, :]
                    elif self.with_tokens:
                        tokens = target[0, 1:-1, :]
                    else:
                        tokens = None
                    if pooling in [n.value for n in list(BasicPoolings)]:
                        if get_hidden:
                            to_pool = target[hidden_layer, 0, 1:-1, :]
                        else:
                            to_pool = target[0, 1:-1, :]
                        embedding = pool_tensor(to_pool, pooling, **kws)
                    elif pooling == "cls" and get_hidden:
                        embedding = target[hidden_layer, 0, 0, :]
                    else:
                        embedding = target[0, 0, :]
                    yield prot, embedding, tokens

            cache.save(df[tkey], fn=esm_official, batch_size=batch_size)
        elif model.endswith("synthyra"):
            key = {
                "esmc_600m_synthyra": "Synthyra/ESMplusplus_large",
                "esmc_300m_synthyra": "Synthyra/ESMplusplus_small",
            }[model]
            m: AutoModelForMaskedLM = AutoModelForMaskedLM.from_pretrained(
                key, trust_remote_code=True
            )
            m.to(device)

            def esmc(proteins):
                if not get_hidden:
                    tmp = m.embed_dataset(
                        tokenizer=m.tokenizer,
                        sequences=proteins,
                        batch_size=batch_size,
                        full_embeddings=True,
                        max_len=2048,
                        pooling_types=[pooling],
                        save=False,
                    )
                    for k, v in tmp.items():
                        if pooling == "cls":
                            pooled = v[0, :]
                        else:
                            pooled = v[1:-1, :].mean(dim=0)
                        yield k, pooled, v
                else:
                    with torch.no_grad():
                        for prot in proteins:
                            tokenized = m.tokenizer(
                                prot, padding=False, return_tensors="pt"
                            ).to(device)
                            output = m(**tokenized, output_hidden_states=True)
                            hidden = output.hidden_states
                            # Each element of hidden is a tensor with shape
                            # (1, sequence_len, dim_model)
                            tokens = (
                                None
                                if not self.with_tokens
                                else hidden[hidden_layer][0, 1:-1, :].cpu()
                            )
                            if pooling == "cls":
                                yield prot, hidden[hidden_layer][0, 0, :].cpu(), tokens
                            else:
                                yield (
                                    prot,
                                    pool_tensor(tokens, pooling, **kws).cpu(),
                                    tokens,
                                )

            cache.save(
                df[tkey],
                fn=esmc,
                batch_size=batch_size * 3,
                # Use fast batching from huggingface
            )
        else:
            raise ValueError(f"esm model {model} not supported")

        return self._finalize_dataset(df, tkey, cache, text_key)

    def _finalize_dataset(
        self,
        dataset_df: pl.DataFrame,
        text_key: str,
        cache: EmbeddingCache,
        cols_remove: str | None = None,
    ) -> Dataset | None:
        if not self.only_cache:
            result = cache.to_dataset(dataset_df, key_col=text_key)
            if cols_remove is not None:
                result = result.remove_columns(cols_remove)
            return result

    def _evo2_embed(
        self,
        dset: Dataset,
        runscript: str,
        text_key: str = "sequence",
        batch_size: int = 10,
        layer: int = 30,
        retries: int = 3,
    ) -> Dataset | None:
        df: pl.DataFrame = dset.to_polars()
        logger.info(f"{df.shape[0]} sequences to embed")
        cache: EmbeddingCache = EmbeddingCache(
            self.workdir,
            save_interval=self.save_interval,
            save_mode=self.save_mode,
            token_prop=self.token_prop,
        )
        error_message = "Evo2 failed to generate predictions"

        def run_evo2(sequences):
            with TemporaryDirectory() as d:
                outdir: Path = Path(d)
                id2seq = dict(zip(range(len(sequences)), sequences))
                fasta = [f">{id}\n{seq}" for id, seq in id2seq.items()]
                input = outdir.joinpath("input.fasta")
                input.write_text("\n".join(fasta))
                token_flag = "" if not self.with_tokens else " --tokens"
                flags = f"--layer {layer} -i {input} -o {outdir}{token_flag}"
                evo_run = run(
                    f"sbatch --wait --parsable {runscript} {flags}",
                    shell=True,
                    capture_output=True,
                )
                jobid = evo_run.stdout.decode().strip()
                outfile = Path(f"slurm-{jobid}.out")
                logger.warning(f"{jobid=}")
                evo_run.check_returncode()
                if outfile.exists():
                    stdout = outfile.read_text()
                    outfile.unlink()
                else:
                    stdout = (
                        "Slurm stdout file not found. Did you delete it by accident?"
                    )
                try:
                    embedding_lf: pl.LazyFrame = pl.scan_parquet(
                        outdir / "pred_input.parquet"
                    )
                    with open(outdir / "seq_idx_map.json", "rb") as f:
                        id_map: dict = json.load(f)
                except FileNotFoundError:
                    logger.debug(
                        f"Evo2 failed to generate predictions\n-- BEGIN STDOUT -- \n{stdout}\n -- END STDOUT --"
                    )
                    raise ValueError(error_message)
                # TODO: need to read this lazily, otherwise your memory is screwed
                # TODO: [2025-12-17 Wed] check that this works now
                id_df: pl.DataFrame = pl.DataFrame(
                    {"seqid": id_map.keys(), "id": id_map.values()}
                )
                joined = embedding_lf.collect().join(id_df, on="id").drop("id")
                embeddings = torch.unbind(joined.select("embeddings").to_torch())
                tokens = (
                    [None] * len(embeddings)
                    if not self.with_tokens
                    else [torch.tensor(t) for t in joined["tokens"]]
                )
                for s, v, t in zip(joined["seqid"], embeddings, tokens):
                    yield s, v, t

        def run_with_retries(sequences):
            # BUG: [2025-12-17 Wed] workaround for the random segfaults you get from the script
            for _ in range(retries):
                try:
                    return run_evo2(sequences)
                except ValueError as e:
                    if e.args[0] == error_message:
                        logger.warning("Retrying...")
                        continue
                    raise e
            logger.critical(f"Failure after {retries} retries")
            raise ValueError(error_message)

        if not retries:
            cache.save(df[text_key], fn=run_evo2, batch_size=batch_size)
        else:
            cache.save(df[text_key], fn=run_with_retries, batch_size=batch_size)

        return self._finalize_dataset(df, text_key, cache)

    def _seqlens_embed(
        self,
        dset: Dataset,
        huggingface: str,
        text_key: str = "sequence",
        model_key: str = "omicseye/seqLens_4096_512_46M-Mp",
        batch_size: int = 64,
        pooling: Literal["cls", "concat"] | BasicPoolings = "mean",
    ) -> Dataset | None:
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
        dataset = dset.add_column("seqlens_uid", list(range(len(dset))))
        df: pl.DataFrame = dataset.to_polars()
        cache: EmbeddingCache = EmbeddingCache(
            self.workdir,
            save_interval=self.save_interval,
            save_mode=self.save_mode,
            token_prop=self.token_prop,
        )
        to_remove = [c for c in dset.column_names if c != "seqlens_uid"]

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        def seqlens(sequences):
            current = dataset.filter(lambda x: x[text_key] in sequences)
            tokenized = current.map(
                lambda x: self._tokenize(
                    x, tokenizer=tokenizer, text_key=text_key, max_length=512
                ),
                batched=True,
                remove_columns=to_remove,
            )
            seqlens_uid2seq = dict(zip(current["seqlens_uid"][:], current[text_key][:]))

            loader = DataLoader(
                tokenized, batch_size=batch_size, collate_fn=data_collator
            )
            for i, batch in enumerate(loader):
                # logger.debug(f"beginning batch {i}")
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                output: MaskedLMOutput = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                attention_mask = attention_mask.to("cpu")
                # logger.debug(f"batch {i} complete")
                last_hidden = output.hidden_states[-1].to("cpu")
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
                for i, (e, seqlens_uid) in enumerate(
                    zip(
                        torch.unbind(embedding, axis=0),
                        torch.unbind(batch["seqlens_uid"]),
                    )
                ):
                    seq = seqlens_uid2seq[seqlens_uid.cpu().item()]
                    # WARNING: you got memory issues from the below line when
                    # hidden_masked[i, :, :] came before None
                    token = None if not self.with_tokens else hidden_masked[i, :, :]
                    yield seq, e, token

        with torch.no_grad():
            cache.save(df[text_key], fn=seqlens, batch_size=batch_size * 3)
        df = df.drop("seqlens_uid")
        return self._finalize_dataset(df, text_key, cache)

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
        utr_amount: tuple[float | int, float | int] | None = None,
        upstream_context: int = 0,
        downstream_context: int = 0,
        id_remap: dict | None = None,
    ):
        """Initialize preprocesser

        Parameters
        ----------
        id_remap : dict
            Optional dictionary to transform id names from the fasta file stem.
            If this is provided, samples that can't be remapped are ignored

        Returns
        -------
        split_method : str
            Method to split sequences longer than `max_length`
            Supported methods
            - bin : Split the sequence at successive intervals equal to `max_length`
            - bakta : Split sequence at
        utr_amount : tuple | None
            The percentage or number of bases of intergenic region to be
            included as part of the UTR, measured
            from the feature start (5' UTR) or end (3' UTR)
            By default, takes half of the region from one gene, leaving the other half for the UTR of
            the downstream/upstream
        upstream_context : int
            Number of upstream bases to include into the embedding, relevant to all
            embedded sequences e.g. chunks of genes
            This is not included if `utr_amount` and the sequence to embed is at the
            feature 5' end
        downstream_context : int
            Number of downstream bases to include into the embedding, relevant to all
            embedded sequences e.g. chunks of genes
            This is not included if `utr_amount` and the sequence to embed is at the
            feature 3' end
        """
        if split_method != "bin" and anno_path is None:
            raise ValueError(
                "`anno_path` file must be provided unless split_rule is `bin`!"
            )
        self.split_method: SPLIT_METHODS = split_method
        self.id_remap: dict | None = id_remap
        self.annotations: Path = anno_path
        self.max_length: int = max_length
        self.utr_amount: tuple[float | int, float | int] | None = utr_amount
        self.upstream_context: int = upstream_context
        self.downstream_context: int = downstream_context
        accepted_suffixes = {".fasta", ".fna", ".fa"}

        self.fastas: list[Path] = [
            s for s in seq_path.iterdir() if s.suffix in accepted_suffixes
        ]

    def _sample_dict(self, sample, record: DNA, **kwargs) -> dict:
        """Boilerplate to create sequence entry for generator"""
        required = {"start", "stop", "seqindex"}
        val = {
            "sample": sample,
            "seqid": record.metadata["id"],
            "sequence": str(record),
            "description": record.metadata["description"],
        }
        val.update(kwargs)
        if (required & val.keys()) != required:
            raise ValueError(
                f"The following keys must be passed in _sample_dict: {required}"
            )
        val["uid"] = f"{sample}_{val['seqid']}:{val['start']}-{val['stop']}"
        return val

    def _process_record(
        self,
        sample: str,
        record: DNA,
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
                start = max(0, acc - self.upstream_context)
                stop = min(len(record), acc + self.max_length + self.downstream_context)
                current = record[start:stop]
                val: dict = self._sample_dict(
                    sample=sample,
                    record=current,
                    seqindex=i,
                    start=start,
                    stop=stop,
                )
                vals.append(val)
                acc += self.max_length
                i += 1
        elif self.split_method == "bakta":
            filtered = anno.filter(pl.col("#Sequence Id") == record.metadata["id"])
            if not filtered.is_empty():
                split_by_length = split_features(
                    filtered,
                    self.max_length,
                    "Start",
                    "Stop",
                    indicate_ends=True,
                    prefix="chunk",
                )
                split_by_length = add_intergenic(
                    record, split_by_length, "Start", "Stop"
                )
                for i, row in enumerate(split_by_length.iter_rows(named=True)):
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
                        # Start, stop indices after (possible) context widening and
                        # splitting into chunks
                        start=indices[0],
                        stop=indices[1],
                        # Actual start, stop indices of features
                        start_actual=row["Start"],
                        stop_actual=row["Stop"],
                    )
                    vals.append(val)
        return vals

    def _get_subsequence(
        self, record: DNA, row: dict, start: str = "Start", stop: str = "Stop"
    ) -> tuple[DNA, tuple[int, int]]:
        start_idx, stop_idx = row[start], row[stop]
        # Get widths of actual downstream, upstream sequences for percentage-based
        # widening. Overriden if widening dowstream, upstream by flat number of bases
        downstream_width, upstream_width = (
            row.get("downstream_intergenic", 0),
            row.get("upstream_intergenic", 0),
        )
        is_5prime, is_3prime = row.get("is_5prime", False), row.get("is_3prime", False)
        upstream, downstream = 0, 0
        add_utrs = self.utr_amount is not None
        if add_utrs:
            # Don't add UTR regions unless the current entry is at a feature boundary
            if is_5prime and isinstance(self.utr_amount[0], float):
                upstream = max(0, math.floor(upstream_width * self.utr_amount[0]))
            elif is_5prime:
                upstream = max(0, self.utr_amount[0])
            if is_3prime and isinstance(self.utr_amount[1], float):
                downstream = max(0, math.floor(downstream_width * self.utr_amount[1]))
            elif is_3prime:
                downstream = max(0, self.utr_amount[1])

        # If adding UTRs, only add more context to gene chunks
        if self.upstream_context and (not is_5prime or not add_utrs):
            upstream += self.upstream_context
        if self.downstream_context and (not is_3prime or not add_utrs):
            downstream += self.downstream_context

        start_idx = max(0, start_idx - upstream)
        stop_idx = min(stop_idx + downstream, len(record))
        return record[start_idx:stop_idx], (start_idx, stop_idx)

    def gen(self):
        """Return generator object, for use with Datasets.from_generator
        Every dict produced by the generator has at least the following keys:
        [sample, seqid, sequence, description, seqindex]
        """
        for fasta in self.fastas:
            id = fasta.stem
            if (self.id_remap is not None) and ((id := self.id_remap.get(id)) is None):
                logger.warning(
                    f"""
                    Fasta file '{fasta.name}' could not be remapped with `id_remap`\n
                    Skipping...
                    """
                )
                continue
            if self.split_method == "bakta":
                try:
                    anno = pl.read_csv(
                        self.annotations.joinpath(f"{id}_bakta.tsv"),
                        separator="\t",
                        skip_rows=5,
                        infer_schema_length=None,
                    ).with_columns(pl.col("#Sequence Id").cast(pl.String))
                except FileNotFoundError:
                    logger.warning(
                        f"bakta file {id}_bakta.tsv not found! Skipping this sample"
                    )
                    anno = None
            else:
                anno = None
            if anno is not None or self.split_method != "bakta":
                for record in skbio.io.read(fasta, "fasta", constructor=DNA):
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
        savepath: Path | None,
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
    ) -> None | Dataset:
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
        dataset: Dataset = Dataset.from_generator(spp.gen)
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
                logger.warning("No sequence metadata provided")
        if savepath is None:
            return dataset
        dataset.save_to_disk(dataset_path=savepath)
