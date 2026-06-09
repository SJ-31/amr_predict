#!/usr/bin/env python3

import os
import pickle
import sys
from collections import defaultdict
from collections.abc import Callable

import numpy as np
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
from amr_predict.cache import EmbeddingCache, LinkedDataset
from amr_predict.embedding import embedding_size
from amr_predict.enums import BasicPoolings, SeqTypes
from amr_predict.evaluation import pami_wrapper, to_binary_form
from amr_predict.models import BaseNN
from amr_predict.sae import BatchTopK
from amr_predict.sae_external import get_default_cfg
from amr_predict.utils import ModuleConfig, read_tabular
from attrs import asdict
from beartype import beartype
from Bio import SeqIO
from datasets import load_from_disk
from datasets.arrow_dataset import Dataset
from loguru import logger
from scipy.cluster.hierarchy import cut_tree
from torch.utils.data import DataLoader

from env import SnakeEnv

if TYPE_CHECKING:
    from snakemake.iocontainers import snakemake

ENV: SnakeEnv = SnakeEnv.new(snakemake.config)
PARAMS: dict = snakemake.params
INPUT = snakemake.input
SCOL = ENV.metadata.sample_col
LABEL_SEP = ENV.metadata.label_sep
LCOL = ENV.metadata.label_col
os.environ["HF_HOME"] = ENV.huggingface

logger.remove(0)
logger.enable("amr_predict")
logger.add(
    PARAMS["log"],
    format=(
        "[<red>{time:HH:mm:ss}</red>] "
        "<yellow>{level}</yellow>: "
        "<cyan>{message}</cyan>"
        "  {extra}"
    ),
    level="TRACE",
)
# * Utility functions


def seqtype_from_params() -> SeqTypes:
    return SeqTypes[PARAMS["seqtype"].upper()]


def pooling_from_params() -> BasicPoolings:
    return BasicPoolings[PARAMS["pooling"].upper()]


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
    with open(file, "rb") as f:
        obj = pickle.load(f)
    return obj["train"], obj["test"], obj["val"]


@beartype
def load_embeddings(
    cache_completion_file: str,
    variation: str,
    vmethod: str,
    embedding_method: str = PARAMS.get("embedding_method"),
    seqtype: Literal["aa", "nuc"] = PARAMS.get("seqtype"),
    level: Literal["tokens", "seqs"] = PARAMS.get("level"),
    dataset_path: Path = ENV.datasets,
    pooling: str = PARAMS.get("pooling", "").upper(),
    with_metadata: bool = False,
) -> LinkedDataset:
    cache_path: Path = Path(cache_completion_file).with_suffix("")
    cache = EmbeddingCache(dir=cache_path)
    seqs: Path = dataset_path / "sequences" / f"{seqtype}-{variation}-{vmethod}"
    seq_df: pl.DataFrame = load_from_disk(seqs).to_polars()
    if ENV.test:
        seq_df = seq_df.with_row_index(name="dummy_id")
    lm = ENV.embedding_methods[SeqTypes[seqtype.upper()]][embedding_method].model
    pooling = BasicPoolings[pooling] if level == "seqs" else None
    if with_metadata and ENV.metadata.file.exists():
        seq_df = seq_df.join(
            read_tabular(ENV.metadata.file),
            how="left",
            left_on="id",
            right_on=ENV.metadata.sample_col,
        )
    elif with_metadata:
        raise ValueError("No existing metadata file specified in env")
    dset = cache.to_dataset(
        df=seq_df,
        key_col="sequence",
        level=level,
        new_col="x",
        max_len=embedding_size(lm),
        subseq_agg=pooling,
    )
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


def collect_from_pkl(result_handler: Callable[[dict], pl.DataFrame]) -> pl.DataFrame:
    """Helper function to aggregate metrics stored in individual
    pickle files into a dataframe.

    Parameters
    ----------
    result_handler : Callable
        Function that receives the pickled dictionary and produces the initial
        result dataframe for each file. This function
        is expected to remove the key containing the results from the dictionary

    Returns
    -------
    DataFrame of results

    Notes
    -----
    This function reads directly from snakemake input, and each file
    must be pickle file containing a dictionary of results and rule metadata
    """
    result = []
    for file in INPUT:
        with open(file, "rb") as f:
            obj: dict = pickle.load(f)
            df: pl.DataFrame = result_handler(obj)
            meta_expr = [pl.lit(k).alias(v) for k, v in obj.items()]
            df = df.with_columns(*meta_expr)
            result.append(df)
    combined = pl.concat(result, how="diagonal_relaxed")
    return combined


# * Rules


def get_activations():
    dset: LinkedDataset = load_embeddings(
        cache_completion_file=INPUT["embeddings"],
        variation="natural",
        vmethod="0",
    )
    if not INPUT["sae"]:
        return from_pretrained()
        # TODO: unfinished
    else:
        sae = lookup_sae(PARAMS["sae"], act_size=dset[0]["x"].shape[1])
        train_idx, test_idx, val_idx = get_dset_indices(INPUT["indices"])
        sae.eval()
        sae.load_state_dict(torch.load(INPUT["sae"]))
        test_portion = dset.select(test_idx)
        predict_on = test_portion["x"][:]
        activations = sae.predict_step(predict_on)
        samples = test_portion["id"][:]
        act_dset = Dataset.from_dict({"id": samples, "activation": activations})
        act_dset.save_to_disk(snakemake.output[0])
        loss_out = {
            it: PARAMS[it]
            for it in ("seqtype", "level", "embedding_method", "pooling", "sae")
        }
        loss_out["loss"] = sae.training_step({sae.x_key: predict_on}, None)
        with open(snakemake.output[1], "wb") as f:
            pickle.dump(loss_out, f)


def sae_label_eval():
    from amr_predict.evaluation import EvalSAE

    metadata = read_tabular(ENV.metadata.file).unique(ENV.metadata.sample_col)
    dataset = load_from_disk(INPUT[0]).with_format("torch", dtype=torch.float32)
    size = dataset["activation"][:].shape[1]
    dataset: pl.DataFrame = dataset.to_polars().cast(
        {"activation": pl.Array(pl.Float32, size)}
    )
    dataset = dataset.join(
        metadata,
        how="left",
        left_on="id",
        right_on=ENV.metadata.sample_col,
        validate="m:1",
    )
    eva = EvalSAE(
        acts=dataset["activation"].to_torch(), threshold=ENV.eval_sae.threshold
    )
    metrics = eva.score_latents(
        labels=dataset.drop("activation"),
        label_col=ENV.metadata.label_col,
        sample_col="id",
        label_sep=ENV.metadata.label_sep,
    )
    result = {
        item: PARAMS[item]
        for item in ("seqtype", "level", "embedding_method", "pooling", "sae")
    }
    result["metrics"] = metrics
    with open(snakemake.output[0], "wb") as f:
        pickle.dump(result, f)


def collect_sae_label_evals():
    import polars.selectors as cs

    combined = collect_from_pkl(
        lambda x: x.pop("metrics")
        .report(k=ENV.eval_sae.top_k, by=ENV.eval_sae.top_k_by)
        .explode(cs.array(), cs.list())
    )  # WARNING: [2026-05-28 Thu] this is exceptionally slow
    # and produces millions of rows if top_k is too high

    combined.write_csv(snakemake.output[0])


def collect_sae_perf():
    combined = collect_from_pkl(lambda x: pl.DataFrame(x.pop("loss")))
    combined.write_csv(snakemake.output[0])


def write_training_indices(token_level: bool):
    if token_level:
        train_idx, test_idx = ms.train_test_split(
            range(
                load_embeddings(
                    cache_completion_file=INPUT[0],
                    variation="natural",
                    vmethod="0",
                ).shape[0]
            ),
            **asdict(ENV.write_training_indices),
        )
    else:
        sample_df: pl.DataFrame = load_from_disk(INPUT[0]).to_polars()
        sample_df = sample_df.with_columns(pl.row_index())
        train_idx, test_idx = ms.train_test_split(
            sample_df["index"], **asdict(ENV.write_training_indices)
        )
    train_idx, val_idx = ms.train_test_split(train_idx, random_state=ENV.rng)
    if not isinstance(train_idx, np.ndarray):
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        test_idx = np.array(test_idx)
    with open(snakemake.output[0], "wb") as f:
        pickle.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)


def write_token_training_indices():
    write_training_indices(True)


def write_seq_training_indices():
    write_training_indices(False)


def train_sae():
    from lightning.pytorch.loggers import WandbLogger

    cache_path: Path = Path(INPUT[0]).with_suffix("")
    rconfig = ENV.train_sae
    train_kws = rconfig.trainer.to_kws()
    sae_name = PARAMS["sae"]
    run_name = (
        f"{cache_path.stem}-{PARAMS['level']}-{PARAMS['seqtype']}-train_sae-{sae_name}"
    )
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
            run_name, project=ENV.wandb_project, save_dir=logdir
        )
        if logdir.exists():
            ckpt_dir = "last"
    dset: LinkedDataset = load_embeddings(
        cache_completion_file=INPUT[0], variation="natural", vmethod="0"
    )
    sae = lookup_sae(sae_name, act_size=dset[0]["x"].shape[1])
    load_kws = rconfig.dataloader.to_kws()
    trainer = L.Trainer(callbacks=[ckpt_callback], **train_kws)
    train_idx, _, val_idx = get_dset_indices(INPUT[1])
    train_l = DataLoader(dset.select(train_idx), **load_kws)
    val_l = DataLoader(dset.select(val_idx), **load_kws)
    trainer.fit(sae, train_dataloaders=train_l, val_dataloaders=val_l, ckpt_path="last")
    torch.save(sae.state_dict(), snakemake.output[0])


def make_seq_dataset():
    from amr_predict.random import Perturber, Randomizer

    dfs: pl.DataFrame = []
    seqtype = seqtype_from_params()
    for spec in ENV.fastas[seqtype]:
        dfs.append(read_fasta(spec.file, spec.header_style))
    combined = pl.concat(dfs)
    for col in ("id", "sequence"):
        if not combined[col].is_duplicated().any():
            logger.warning(f"Duplicate {col} present in fastas")
            dups: pl.DataFrame = combined.filter(combined[col].is_duplicated())
            dups.write_csv(ENV.outdir / f"duplicated_{col}_{seqtype.value}.csv")
            combined = combined.unique(col)
    if ENV.test:
        combined = combined.sample(n=1000)
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
    label_df: pl.DataFrame = read_tabular(INPUT[0]).unique()
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
        yaml.safe_dump(pattern_stats, f)


def embedding_metrics():
    analysis_group = PARAMS["analysis"]
    assert analysis_group in {"nn", "corr"}
    run_kws = {}
    if analysis_group == "nn":
        from amr_predict.metrics import NeighborMetrics

        obj_fn = NeighborMetrics
        kws = asdict(ENV.neighbor_metrics)
        run_kws["with_randomization"] = True
    else:
        from amr_predict.metrics import EmbeddingCorrelations

        obj_fn = EmbeddingCorrelations
        kws = asdict(ENV.embedding_correlations)
        kws["level"] = PARAMS["level"]

    dset = load_embeddings(
        cache_completion_file=INPUT[0],
        variation="natural",
        vmethod="0",
        with_metadata=True,
    )
    kws["seed"] = ENV.rng
    obj = obj_fn(dset, **kws)
    test_results, dist = obj.run(**run_kws)
    metadata_expr = []
    dist_dict = {"result": dist}
    for item in ("seqtype", "level", "embedding_method", "pooling", "analysis"):
        metadata_expr.append(pl.lit(PARAMS[item]).alias(item))
        dist_dict[item] = PARAMS[item]
    test_results = test_results.with_columns(*metadata_expr)
    test_results.write_csv(snakemake.output[0])
    with open(snakemake.output[1], "wb") as f:
        pickle.dump(dist_dict, f)


def collect_embedding_metrics():
    corr_all, nn_all, dist_dfs = [], [], []
    for path in (Path(i) for i in INPUT):
        if path.suffix == ".csv":
            df: pl.DataFrame = pl.read_csv(path)
            if (df["analysis"] == "nn").all():
                append_to = nn_all
            else:
                append_to = corr_all
        else:
            append_to = dist_dfs
            with open(path, "rb") as f:
                dist = pickle.load(f)
                df = pl.concat(
                    [
                        pl.DataFrame({"metric": k, "value": v})
                        for k, v in dist["result"].items()
                    ],
                    how="diagonal_relaxed",
                ).with_columns(
                    *[pl.lit(v).alias(k) for k, v in dist.items() if k != "result"]
                )
        append_to.append(df)
    pl.concat(nn_all, how="diagonal_relaxed").write_csv(snakemake.output["nn_all"])
    pl.concat(corr_all, how="diagonal_relaxed").write_csv(snakemake.output["corr_all"])

    # The distribution stores embedding distributions
    # TODO: add graphing and testing to compare distributions
    Path(snakemake.output["nn_comparison"]).touch()


def probing_permutation_tests():
    from amr_predict.evaluation import Evaluator
    from amr_predict.metadata import encode_strs
    from amr_predict.metrics import classifier_dispatch

    # Parallelize over different models, tasks, and embeddings
    # with 'method' as the model key
    # and `variation` as the task name
    model_name = PARAMS["method"]
    kws: dict = ENV.probing.classifiers[model_name]
    model = classifier_dispatch(model_name, **kws)
    task = (PARAMS["variation"],)
    eva = Evaluator(model=model, x_key="x", task_type="classification", seed=ENV.rng)
    dset = Dataset.from_polars(load_embeddings(INPUT[0]).to_pl().select(("x",) + task))
    dset, encoder = encode_strs(dset, task_names=task)
    t1 = eva.permutation_test(dset, "class_label").with_columns(
        pl.lit("permutation_test_1").alias("test")
    )
    t2 = eva.permutation_test(dset, "per_class_feature").with_columns(
        pl.lit("permutation_test_2").alias("test")
    )
    combined = pl.concat([t1, t2], how="vertical_relaxed")
    combined.write_csv(snakemake.output[0])


# TODO: Will need to adjust for multiple testing with this
# def collect_probing


def find_baseline():
    from amr_predict.metrics import PerturbationMetrics

    metadata_expr = [
        pl.lit(PARAMS[item]).alias(item)
        for item in ("seqtype", "level", "embedding_method", "pooling")
    ]

    dfs = []
    for cls_name, kws in ENV.find_baseline.classifiers.items():
        M: PerturbationMetrics = PerturbationMetrics(
            id_col=ENV.perturbation_metrics.id_col,
            embedding_distance=ENV.perturbation_metrics.embedding_distance,
            natural=load_embeddings(INPUT["natural"], "natural", "0"),
            random=None,
            perturbed=None,
            level=PARAMS["level"],
            seed=ENV.rng,
            classifier_name=cls_name,
            classifier_kws=(kws or {}),
        )
        res = M.find_baseline(
            n_repeats=ENV.find_baseline.n_repeats, split_kws=ENV.find_baseline.split_kws
        ).with_columns(*metadata_expr)
        dfs.append(res)
    pl.concat(dfs).write_csv(snakemake.output[0])


def collect_baseline():
    dfs = [
        pl.read_csv(input).with_columns(pl.col("pooling").cast(pl.String))
        for input in INPUT
    ]
    pl.concat(dfs, how="diagonal_relaxed").write_csv(snakemake.output[0])


def perturbation_metrics():
    from amr_predict.metrics import PerturbationMetrics

    rand_method = PARAMS["rand_method"]
    if ENV.test:
        ENV.perturbation_metrics.id_col = "dummy_id"
    M: PerturbationMetrics = PerturbationMetrics(
        natural=load_embeddings(INPUT["natural"], "natural", "0"),
        perturbed=load_embeddings(INPUT["perturbed"], "perturbed", PARAMS["method"]),
        random=load_embeddings(INPUT["random"], "randomized", rand_method),
        level=PARAMS["level"],
        seed=ENV.rng,
        random_is_pairable="denovo" not in rand_method,
        **asdict(ENV.perturbation_metrics),
    )
    meta = {k: PARAMS[k] for k in ("seqtype", "level", "pooling", "embedding_method")}
    result: dict = {
        "perturbation": PARAMS["method"],
        "randomization": rand_method,
        "result": M.run(),
    }
    result = result | meta
    with open(snakemake.output[0], "wb") as f:
        pickle.dump(result, f)


def collect_perturbation_metrics():
    to_combine = defaultdict(list)
    for file in INPUT:
        with open(file, "rb") as f:
            data: dict = pickle.load(f)
        meta_expr = [pl.lit(v).alias(k) for k, v in data.items() if k != "result"]
        for k, v in data["result"].items():
            v: pl.DataFrame
            to_combine[k].append(v.with_columns(*meta_expr))
    for k, v in to_combine.items():
        df = pl.concat(v)
        df.write_csv(snakemake.output[k])


# def label_clustering():
#     label_df = read_tabular(INPUT[0]).unique()
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
    from amr_predict.cache import expand_max_len
    from amr_predict.embedding import EmbeddingModels, ModelEmbedder

    df: pl.DataFrame = load_from_disk(INPUT[0]).to_polars()
    spec = ENV.embedding_methods[seqtype_from_params()][PARAMS["embedding_method"]]
    model: EmbeddingModels = spec.model
    max_length = embedding_size(model)
    df = expand_max_len(df, max_len=max_length, seq_col="sequence")
    kws: dict = spec.kws
    out = Path(snakemake.output[0])
    cache_path = out.with_suffix("")
    if not cache_path.exists():
        cache_path.mkdir()
    kws["workdir"] = cache_path
    kws["huggingface"] = ENV.huggingface
    kws["save_mode"] = PARAMS["level"]
    if PARAMS["level"] == "seqs":
        kws["pooling"] = pooling_from_params()
        kws["pooling_kws"] = spec.poolings[pooling_from_params()] or {}
    else:
        kws["pooling"] = None
        kws["pooling_kws"] = {}
    kws["save_proba"] = PARAMS["level"] == "tokens"
    embedder: ModelEmbedder = ModelEmbedder.new(model, only_cache=True, **kws)
    embedder.embed(dataset=Dataset.from_polars(df))
    out.write_text("completed")


# * Entry

with logger.contextualize(
    rule=snakemake.rule,
    outs=[f"{Path(o).parent.name}/{Path(o).name}" for o in snakemake.output],
):
    if rule_fn := globals().get(snakemake.rule):
        rule_fn()
