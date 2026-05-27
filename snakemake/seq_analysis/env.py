#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Union

import amr_predict.enums as ae
import cattrs
import yaml
from attr.validators import instance_of
from attrs import Factory, asdict, define, field, validators
from snakemake.io import expand, multiext
from yte import process_yaml

Levels = Enum("Levels", (("TOKENS", "tokens"), ("SEQS", "seqs")))


cattrs.register_structure_hook(Union[str, bool], Union[str, bool])


@define
class Metadata:
    file: Path = field(converter=Path)
    sample_col: str
    label_col: str
    label_sep: str


@define
class TrainerCfg:
    # ── Core training ──────────────────────────────────────────────
    max_epochs: int = field(default=100)
    min_epochs: Optional[int] = field(default=None)
    max_steps: int = field(default=-1)  # -1 = no limit
    min_steps: Optional[int] = field(default=None)

    # ── Hardware ───────────────────────────────────────────────────
    accelerator: str = field(default="auto")  # "cpu", "gpu", "tpu", "auto"
    devices: str = field(default="auto")
    num_nodes: int = field(default=1)
    strategy: str = field(default="auto")  # "ddp", "fsdp", "deepspeed", etc.
    precision: str = field(default="32-true")  # "16-mixed", "bf16-mixed"

    # ── Gradient / optimisation ────────────────────────────────────
    accumulate_grad_batches: int = field(default=1)
    gradient_clip_val: Optional[float] = field(default=None)
    gradient_clip_algorithm: str = field(default="norm")  # "norm" | "value"

    # ── Validation & checkpointing ─────────────────────────────────
    val_check_interval: float = field(default=1.0)
    check_val_every_n_epoch: int = field(default=1)
    num_sanity_val_steps: int = field(default=2)
    log_every_n_steps: int = field(default=50)

    # ── Reproducibility ────────────────────────────────────────────
    deterministic: Union[bool, str] = field(default=False)  # True | "warn"

    # ── Misc ───────────────────────────────────────────────────────
    fast_dev_run: bool = field(default=False)
    overfit_batches: float = field(default=0.0)
    limit_train_batches: Optional[float] = field(default=1.0)
    limit_val_batches: Optional[float] = field(default=1.0)
    limit_test_batches: Optional[float] = field(default=1.0)
    limit_predict_batches: Optional[float] = field(default=1.0)
    enable_progress_bar: bool = field(default=True)
    enable_model_summary: bool = field(default=True)
    default_root_dir: Optional[Path] = field(default=None)

    # ── Validators ─────────────────────────────────────────────────
    @max_epochs.validator
    def _check_max_epochs(self, attribute, value):
        if value < 1:
            raise ValueError("max_epochs must be >= 1")

    @accumulate_grad_batches.validator
    def _check_accum(self, attribute, value):
        if value < 1:
            raise ValueError("accumulate_grad_batches must be >= 1")

    def to_kws(self) -> dict:
        return asdict(self)


@define
class DataLoaderCfg:
    # ── Core ───────────────────────────────────────────────────────
    batch_size: int = field(default=32)
    shuffle: bool = field(default=False)
    num_workers: int = field(default=0)
    pin_memory: bool = field(default=False)
    drop_last: bool = field(default=False)

    # ── Collation & prefetching ────────────────────────────────────
    prefetch_factor: Optional[int] = field(
        default=None
    )  # only active if num_workers > 0
    persistent_workers: bool = field(default=False)  # keep workers alive between epochs

    # ── Timeout & multiprocessing ──────────────────────────────────
    timeout: float = field(default=0.0)  # 0 = no timeout
    multiprocessing_context: Optional[Any] = field(
        default=None
    )  # "fork", "spawn", "forkserver"

    # ── Memory ─────────────────────────────────────────────────────
    pin_memory_device: str = field(
        default=""
    )  # e.g. "cuda:0"; only used if pin_memory=True

    # ── Validators ─────────────────────────────────────────────────
    @batch_size.validator
    def _check_batch_size(self, attribute, value):
        if value < 1:
            raise ValueError("batch_size must be >= 1")

    @num_workers.validator
    def _check_num_workers(self, attribute, value):
        if value < 0:
            raise ValueError("num_workers must be >= 0")

    @timeout.validator
    def _check_timeout(self, attribute, value):
        if value < 0:
            raise ValueError("timeout must be >= 0")

    @prefetch_factor.validator
    def _check_prefetch_factor(self, attribute, value):
        if value is not None and value < 1:
            raise ValueError("prefetch_factor must be >= 1")

    def to_kws(self) -> dict:
        """Unpack directly into DataLoader(dataset, **cfg.to_dataloader_kwargs())."""
        return asdict(self, filter=lambda a, v: v is not None or a.default is None)


@define
class EmbeddingMethod:
    model: ae.EmbeddingModels
    poolings: dict[ae.BasicPoolings, dict[str, Any] | None] = field(factory=dict)
    kws: dict[str, Any] | None = field(factory=dict)


@define
class LabelClustering:
    min_height: float
    method: str = field(
        default="average", validator=validators.in_(["single", "average", "complete"])
    )
    metric: str = "jaccard"


@define
class TrainSae:
    trainer: TrainerCfg
    dataloader: DataLoaderCfg


def needs_val_if_not_custom(inst, attr, val, allowed: Enum | None = None):
    if not allowed:
        return inst.source == "custom" or (inst.source != "custom" and val)
    return inst.source == "custom" or (inst.source != "custom" and val in allowed)


@define
class SaeCfg:
    source: str | None = None
    variant: str = field(default="BatchTopK", validator=validators.in_(["BatchTopK"]))
    kws: dict[str, Any] = field(factory=dict)
    level: Levels | None = field(default=None, validator=needs_val_if_not_custom)
    embedding: str | None = field(default=None, validator=needs_val_if_not_custom)

    @level.validator
    def check_level(inst, attr, val):
        return needs_val_if_not_custom(inst, attr, val, Levels)


@define
class EvalSAECfg:
    threshold: float
    top_k: int
    top_k_by: str = "activation_prop"
    # drop: dict  = field(default )


@define
class WriteTrainingIndices:
    test_size: float = 0.4
    shuffle: bool = True


@define
class FastaSpec:
    file: str
    header_style: Literal["uniprot"] = "uniprot"


# TODO: there's gotta be a better pattern for this... though it doesn't take that long to write


@define
class Perturbation:
    method: ae.Perturbations
    seqtype: ae.SeqTypes
    kws: ae.PerturbationCfg = field(
        converter=lambda x: x or ae.PerturbationCfg(), factory=ae.PerturbationCfg
    )


@define
class Randomization:
    method: ae.RandomizationMethods
    kws: dict[str, Any] = field(converter=lambda x: x or {}, factory=dict)


@define
class SequenceVariants:
    perturbed: dict[str, Perturbation] | None = None
    random: dict[str, Randomization] | None = None


@define
class NeighborMetricsCfg:
    category_cols: list[str] | None = None
    anno_cols: list[str] | None = None
    subsample: float | None = None
    n_neighbors: int = 5
    nn_kws: dict = field(factory=dict)
    n_resample: int = 5_000
    anno_sep: str = ";"


@define
class PerturbationMetricsCfg:
    id_col: str = "id"
    embedding_distance: Literal["cosine", "euclidean", "manhattan"] = "cosine"
    classifier_name: str = "LogisticRegression"
    cv: int = 5
    n_distance_resampling: int = 5
    rns_kws: dict = field(factory=lambda: {"prop": 1, "iterations": 20})
    subsample_prop: dict | None = field(factory=dict)


@define
class EmbeddingCorrelationsCfg:
    columns: dict[str, ae.SeqCovariates]
    embedding_distance: Literal["cosine", "euclidean", "manhattan"]
    sequence_distance: str = "kmer"
    anno_sep: str = ";"
    seed: int | None = None
    n_resample: int = 10_000
    tree_file: Path | None = field(
        default=None, converter=lambda x: Path(x) if x else x
    )


@define
class FindBaseline:
    classifiers: dict[str, dict | None]
    n_repeats: int = 5
    split_kws: dict | None = field(factory=dict)


@define
class SnakeEnv:
    huggingface: str
    rng: int
    outdir: Path = field(converter=Path)
    metadata: Metadata
    sequence_variants: SequenceVariants
    resources: dict = field(validator=instance_of(dict))
    saes: dict[Literal["custom", "pretrained"], dict[str, SaeCfg] | None] = field(
        validator=instance_of(dict)
    )
    fastas: dict[ae.SeqTypes, list[FastaSpec]] = field(
        validator=validators.deep_mapping(key_validator=instance_of(ae.SeqTypes))
    )
    embedding_methods: dict[ae.SeqTypes, dict[str, EmbeddingMethod]] = field(
        validator=validators.deep_mapping(
            key_validator=instance_of(ae.SeqTypes),
        )
    )

    # Rules
    train_sae: TrainSae
    write_training_indices: WriteTrainingIndices
    label_clustering: LabelClustering
    neighbor_metrics: NeighborMetricsCfg | None
    embedding_correlations: EmbeddingCorrelationsCfg | None
    perturbation_metrics: PerturbationMetricsCfg | None
    find_baseline: FindBaseline | None
    eval_sae: EvalSAECfg

    # Misc rule config
    slurm_time_limit: str = "18-00:00:00"
    co_occurence_min_support: float = 0.3
    save_token_proportion: float = 0.3
    log_wandb: bool = True
    wandb_project: str | None = None
    embedding_key: str = "x"
    embedding_max_lengths: dict = field(default={"esm": 2048, "seqLens": 512})
    test: bool = False
    seqtypes: list[ae.SeqTypes] = field(
        init=False,
        default=Factory(
            lambda self: [
                st
                for st in ae.SeqTypes
                if st in self.fastas and st in self.embedding_methods
            ],
            takes_self=True,
        ),
    )

    @property
    def datasets(self) -> Path:
        return self.outdir / "datasets"

    def get_outputs(self) -> tuple[list, dict]:
        out: list = [
            self.outdir / "label_cooccurrence.csv",
            self.outdir / "cooccurrence_stats.yaml",
            self.outdir / "analyses/nn_all.csv",
            self.outdir / "analyses/covariate_correlation_all.csv",
            self.outdir / "analyses/nn_comparison.csv",
            self.outdir / "analyses/classifier_random_perf.csv",
            self.outdir / "analyses/sae_label_evaluation.csv",
            self.outdir / "analyses/classifiability.csv",
            self.outdir / "analyses/random_neighbor_score.csv",
            self.outdir / "analyses/distance_correlation.csv",
        ]
        custom_saes: dict = self.saes["custom"]

        for st in self.seqtypes:
            # TODO: the intermediate outputs (mainly embeddings) can be omitted
            # and left to snakemake wildcards to decide
            # The final outputs should be the analyses which will specify which
            # embeddings are required
            acts_prefix: str = f"{self.datasets}/activations_{st.value}"
            embedding_prefix: str = f"{self.datasets}/embedded_{st.value}"

            for mname, mspec in self.embedding_methods[st].items():
                for s in custom_saes.keys():
                    out.append(f"{acts_prefix}_tokens/{mname}-0-{s}")
                for p in mspec.poolings:
                    out.append(
                        f"{embedding_prefix}_seqs/natural-0/{mname}-{p.value}.completed"
                    )

                    for ptb, ptb_spec in self.sequence_variants.perturbed.items() or {}:
                        if ptb_spec.seqtype == st:
                            out.append(
                                f"{embedding_prefix}_seqs/perturbed-{ptb}/{mname}-{p.value}.completed"
                            )
                    for rnd in self.sequence_variants.random.keys() or ():
                        out.append(
                            f"{embedding_prefix}_seqs/randomized-{rnd}/{mname}-{p.value}.completed"
                        )
                    for sae in custom_saes:
                        out.append(f"{acts_prefix}_seqs/{mname}-{p.value}-{sae}")

            if self.saes["pretrained"]:
                for sae, spec in self.saes["pretrained"].items():
                    out.append(
                        self.datasets
                        / f"activations_{st.value}_{spec.level.value}"
                        / f"{spec.embedding}-_-{sae}"
                    )

        return out

    @classmethod
    def new(cls, data: str | dict, with_yte: bool = True) -> SnakeEnv:
        if isinstance(data, str):
            assert Path(data).exists() and data.endswith(
                ".yaml"
            ), "Must pass a yaml file"
            with open(data, "r") as f:
                data = process_yaml(f) if with_yte else yaml.safe_load(f)
        return cattrs.structure(data, SnakeEnv)

    def to_dict(self) -> dict:
        return asdict(self)


def test_cfg():
    from pyhere import here

    file = here("snakemake", "seq_analysis", "env.yaml")
    cfg = SnakeEnv.new(str(file))
    cfg.get_outputs()
    return cfg


def expand_nested(
    n1: dict | Sequence, n2: Sequence | None = None, prefix: str = "", **separators
) -> list[str]:
    """Flatten a nested dictionary of keys and values while respecting
    the dictionary structure

    Parameters
    ----------
    separators : kwargs
        Separator characters to use at each nesting level, defined using the format
        _<LEVEL> = SEPARATOR e.g. _2 = '/' to use a slash at level 2
        Defaults to underscore at all levels if not provided
    prefix : str
        Initial prefix to place before files

    Returns
    -------

    Suppose
    combos = {
    "a": [1, 2, 3],
    "b": [5, 6],
    "c": {"c.1": [100, 200], "c.2": [588, 963]},
    }
    expand_nested(combos, prefix = "pref") returns
    [
        "pref_a_1",
        "pref_a_2",
        "pref_a_3",
        "pref_b_5",
        "pref_b_6",
        "pref_c_c.1_100",
        "pref_c_c.1_200",
        "pref_c_c.2_588",
        "pref_c_c.2_963",
    ]
    """
    if separators:
        assert all(
            [
                k.startswith("_")
                and len(k) > 1
                and k.count("_") == 1
                and k.split("_")[1].isdigit()
                for k in separators
            ]
        ), "Separators must use the format _<LEVEL> = SEPARATOR"
    level2sep: dict[int, str] = {int(k.split("_")[1]): v for k, v in separators.items()}
    res = []

    def rec(level: int, acc: str, k: str, v: dict | Sequence) -> list[str]:
        s1 = level2sep.get(max(level - 1, 0), "_")
        s2 = level2sep.get(level, "_")
        pref = f"{acc}{s1}{k}" if acc else k
        if not isinstance(v, dict):
            return [f"{pref}{s2}{item}" for item in v]
        result = []
        for child_key, child_items in v.items():
            tmp = rec(level + 1, pref, child_key, child_items)
            result.extend(tmp)
        return result

    if isinstance(n1, dict):
        nesting = n1
    elif isinstance(n1, Sequence) and isinstance(n2, Sequence):
        assert len(n1) == len(n2), "Keys and values must be the same length"
        nesting = dict(zip(n1, n2))
    else:
        raise ValueError("Argument not recognized")
    for k, v in nesting.items():
        res.extend(rec(0, acc=prefix, k=k, v=v))
    return res
