#!/usr/bin/env python3

import json
import os
import sys

from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.append("/py_lib")

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import fastcluster
import lightning as L
import polars as pl
import sklearn.model_selection as ms
import torch
import yaml
from amr_predict.cache import EmbeddingCache, LinkedDataset, expand_max_len
from amr_predict.embedding import EmbeddingModels, ModelEmbedder, embedding_size
from amr_predict.enums import BasicPoolings, SeqTypes
from amr_predict.evaluation import pami_wrapper, to_binary_form
from amr_predict.models import BaseNN
from amr_predict.pooling import pool_tensor
from amr_predict.random import Perturber, Randomizer
from amr_predict.sae import BatchTopK
from amr_predict.sae_external import get_default_cfg
from amr_predict.utils import ModuleConfig, read_tabular
from attrs import asdict
from beartype import beartype
from Bio import SeqIO
from datasets import load_from_disk
from datasets.arrow_dataset import Dataset
from datasets.packaged_modules import text
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from scipy.cluster.hierarchy import cut_tree
from torch.utils.data import DataLoader

from env import SnakeEnv

if TYPE_CHECKING:
    from snakemake.iocontainers import snakemake

logger.add(sys.stdout, format="{extra}")

ENV: SnakeEnv = SnakeEnv.new(snakemake.config)

PARAMS: dict = snakemake.params

SCOL = ENV.metadata.sample_col
LABEL_SEP = ENV.metadata.label_sep
LCOL = ENV.metadata.label_col


os.environ["HF_HOME"] = ENV.huggingface

# * Utility functions


@beartype
def read_fasta(file: str, header_style: Literal["uniprot"]) -> pl.DataFrame:
    tmp = {"id": [], "sequence": []}
    for record in SeqIO.parse(file, "fasta"):
        if header_style == "uniprot":
            try:
                id = record.id.split("|")[1]
            except IndexError:
                raise ValueError(
                    f"FASTA entry {record} does not have a UniProt-style header"
                )
        tmp["id"].append(id)
        tmp["sequence"].append(str(record.seq))
    df: pl.DataFrame = pl.DataFrame(tmp)
    if not df.select("id").is_duplicated().any():
        logger.warning(f"Fasta file {file} has duplicate ids")
        df = df.unique("id")
    return df


def get_dset_indices(file) -> tuple[list, list, list]:
    """Read saved indices file and return a tuple of train, test, val indices"""
    with open(file, "r") as f:
        obj = json.load(f)
    return obj["train"], obj["test"], obj["val"]


def load_embeddings(
    cache_completion_file: str,
    seqtype: str,
    variation: str,
    vmethod: str,
    embedding_method: str,
    level: Literal["tokens", "seqs"],
    dataset_path: Path = ENV.datasets,
) -> LinkedDataset:
    cache_path: Path = Path(cache_completion_file).with_suffix("")
    cache = EmbeddingCache(dir=cache_path)
    seqs: Path = dataset_path / "sequences" / f"{seqtype}-{variation}-{vmethod}"
    seq_df: pl.DataFrame = load_from_disk(seqs).to_polars()
    lm = ENV.embedding_methods[seqtype][embedding_method].model
    seq_df = account_for_max_length(seq_df, max_len=embedding_size(lm), cache=cache)
    dset = cache.to_dataset(df=seq_df, key_col="sequence", level=level, new_col="x")
    assert isinstance(dset, LinkedDataset)
    return dset


def from_pretrained(dset: LinkedDataset):
    pass


def lookup_sae(spec: str, act_size: int) -> BaseNN:
    from_env = ENV.saes["custom"][spec]
    sae_cfg = get_default_cfg()
    from_env.kws["device"] = "gpu" if torch.cuda.is_available() else "cpu"
    from_env.kws["act_size"] = act_size
    from_env.kws["dtype"] = torch.get_default_dtype()
    variant = from_env.variant
    if variant == "BatchTopK":
        cls = BatchTopK
    else:
        raise NotImplementedError("SAE variant not recognized")
    sae_cfg.update(from_env.kws)
    cfg: ModuleConfig = ModuleConfig(**sae_cfg)
    return cls(cfg=cfg, x_key="x")


# * Rules


def get_activations():
    dset: LinkedDataset = load_embeddings(
        cache_completion_file=snakemake.input["embeddings"],
        variation="natural",
        vmethod="0",
        level=PARAMS["level"],
        seqtype=PARAMS["seqtype"],
    )
    if not snakemake.input["sae"]:
        return from_pretrained()
    # TODO: unfinished
    pass


def write_training_indices():
    sample_df: pl.DataFrame = load_from_disk(snakemake.input[0]).to_polars()
    if PARAMS["level"] == "tokens":
        sample_df = sample_df.with_columns(pl.col("sequence").str.split("")).explode(
            "sequence"
        )
    sample_df = sample_df.with_columns(pl.row_index())
    train_idx, test_idx = ms.train_test_split(
        sample_df["index"], **asdict(ENV.write_training_indices)
    )
    train_idx, val_idx = ms.train_test_split(train_idx, random_state=ENV.rng)
    with open(snakemake.output[0], "w") as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)


def train_sae():
    cache_path: Path = Path(snakemake.input[0]).with_suffix("")
    rconfig = ENV.train_sae
    train_kws = rconfig.trainer.to_kws()
    sae_name = PARAMS["sae"]
    run_name = f"{cache_path.stem}-seq_analysis-train_sae-{sae_name}"
    outpath = Path(snakemake.output[0])
    ckpt_dir = outpath.parent / f".{outpath.name.removesuffix(".pt")}_checkpoints"
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_epochs=5,
        enable_version_counter=True,
        verbose=True,
    )
    if ENV.log_wandb:
        logdir = outpath.parent / f".{outpath.name.removesuffix(".pt")}_wandb"
        train_kws["logger"] = WandbLogger(
            run_name, project="amr_predict", save_dir=logdir
        )
        if logdir.exists():
            ckpt_dir = "last"
    seq_level = PARAMS["level"]
    dset: LinkedDataset = load_embeddings(
        cache_completion_file=snakemake.input[0],
        variation="natural",
        vmethod="0",
        seqtype=PARAMS["seqtype"],
        embedding_method=PARAMS["embedding_method"],
        level=seq_level,
    )
    sae = lookup_sae(sae_name, act_size=dset[0]["x"].shape[1])
    load_kws = rconfig.dataloader.to_kws()
    trainer = L.Trainer(callbacks=[ckpt_callback], **train_kws)
    train_idx, _, val_idx = get_dset_indices(snakemake.input[1])
    train_l = DataLoader(dset.select(train_idx), **load_kws)
    val_l = DataLoader(dset.select(val_idx), **load_kws)
    trainer.fit(sae, train_dataloaders=train_l, val_dataloaders=val_l, ckpt_path="last")
    torch.save(sae.state_dict(), snakemake.output[0])


def pooling_comparison(dataset: str):
    dset: LinkedDataset = load_embeddings(
        snakemake.input[0],
        dataset_path=ENV.datasets,
        variation=PARAMS["variation"],
        vmethod=PARAMS["vmethod"],
        level=PARAMS["level"],
    )
    metadata = dset.meta


def make_seq_dataset():
    dfs: pl.DataFrame = []
    seqtype = SeqTypes[PARAMS["seqtype"].upper()]
    for spec in ENV.fastas[seqtype]:
        dfs.append(read_fasta(spec.file, spec.header_style))
    combined = pl.concat(dfs)
    for col in ("id", "sequence"):
        if not combined[col].is_duplicated().any():
            logger.warning(f"Duplicate {col} present in fastas")
            dups: pl.DataFrame = combined.filter(combined[col].is_duplicated())
            dups.write_csv(ENV.outdir / f"duplicated_{col}_{seqtype.value}.csv")
            combined = combined.unique(col)
    if PARAMS["variation"] == "randomized":
        lookup = ENV.sequence_variants.random[PARAMS["method"]]
        rnd = Randomizer(method=lookup.method, seqtype=seqtype, **lookup.kws)
        combined = rnd.randomize(combined)
    elif PARAMS["variation"] == "perturbed":
        lookup = ENV.sequence_variants.perturbed[PARAMS["method"]]
        ptb = Perturber.new(lookup.method, seqtype=seqtype, cfg=lookup.kws)
        combined = ptb.perturb(combined)
    dset: Dataset = Dataset.from_polars(combined)
    dset.save_to_disk(dataset_path=snakemake.output[0])


def label_cooccurrence():
    from PAMI.frequentPattern.basic.FPGrowth import FPGrowth

    cuda_apriori_available = False
    cudaAprioriTID = None
    try:
        from PAMI.frequentPattern.cuda import cudaAprioriTID

        cuda_apriori_available: bool = True
    except (ModuleNotFoundError, ImportError):
        pass
    if torch.cuda.is_available() and cuda_apriori_available:
        alg = cudaAprioriTID
    else:
        alg = FPGrowth
    label_df: pl.DataFrame = read_tabular(snakemake.input[0]).unique()
    frequent_patterns, pattern_stats = pami_wrapper(
        label_df,
        alg,
        label_col=LCOL,
        sep=LABEL_SEP,
        min_sup=ENV.co_occurence_min_support,
        tmp_file=PARAMS["tmp_file"],
    )
    frequent_patterns.with_columns(pl.col("Patterns").str.join(";")).write_csv(
        snakemake.output[0]
    )
    with open(snakemake.output[1], "w") as f:
        yaml.safe_dump(f, pattern_stats)


# def label_clustering():
#     label_df = read_tabular(snakemake.input[0]).unique()
#     binary = to_binary_form(label_df, sample_col=SCOL, label_col=LCOL, sep=LABEL_SEP)
#     labels = binary.columns
#     linkage = fastcluster.linkage_vector(
#         binary.to_numpy(),
#         metric=RCONFIG["metric"],
#         method=RCONFIG["method"],
#     )
#     cut = cut_tree(linkage, height=RCONFIG["height"])
#     # TODO: should optimize this somehow. Maybe silhouette score? Or dynamically cutting
#     # the tree
#     # Cluster robustness may also work, but you're not confident that the R version will
#     # scale to the size
#     # TODO: also need to put out the stats
#     pl.DataFrame({"label": labels, "cluster": cut.flatten()}).write_csv(smk.output[0])


def get_embeddings():
    df: pl.DataFrame = load_from_disk(snakemake.input[0]).to_polars()
    seqtype = SeqTypes[PARAMS["seqtype"].upper()]
    spec = ENV.embedding_methods[seqtype][PARAMS["embedding_method"]]
    model: EmbeddingModels = spec.model
    max_length = embedding_size(model)
    df = account_for_max_length(df, max_len=max_length, seq_col="sequence", id_col="id")
    kws: dict = spec.kws
    out = Path(snakemake.output[0])
    cache_path = out.with_suffix("")
    if not cache_path.exists():
        cache_path.mkdir()
    kws["workdir"] = cache_path
    kws["huggingface"] = ENV.huggingface
    kws["save_mode"] = PARAMS["level"]
    embedder: ModelEmbedder = ModelEmbedder.new(model, only_cache=True, **kws)
    embedder.embed(dataset=Dataset.from_polars(df))
    out.write_text("completed")


# * Entry

if rule_fn := globals().get(snakemake.rule):
    rule_fn()
