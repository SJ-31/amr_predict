#!/usr/bin/env python3

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Union

import amr_predict.enums as ae
import cattrs
import yaml
from attr.validators import instance_of
from attrs import asdict, define, field, validators
from snakemake.io import expand
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
class WriteTrainingIndices:
    test_size: float = 0.4
    shuffle: bool = True


@define
class FastaSpec:
    file: str
    header_style: Literal["uniprot"] = "uniprot"


@define
class SnakeEnv:
    huggingface: str
    rng: int
    outdir: Path = field(converter=Path)
    metadata: Metadata
    resources: dict = field(validator=instance_of(dict))
    saes: dict[Literal["custom", "pretrained"], dict[str, SaeCfg]] = field(
        validator=instance_of(dict)
    )
    fastas: dict[SeqTypes, list[FastaSpec]] = field(
        validator=validators.deep_mapping(key_validator=instance_of(SeqTypes))
    )
    embedding_methods: dict[SeqTypes, dict[str, EmbeddingMethod]] = field(
        validator=validators.deep_mapping(
            key_validator=instance_of(SeqTypes),
        )
    )

    # Rules
    train_sae: TrainSae
    write_training_indices: WriteTrainingIndices
    label_clustering: LabelClustering

    slurm_time_limit: str = "18-0:0:0"
    co_occurence_min_support: float = 0.3
    save_token_proportion: float = 0.3
    log_wandb: bool = True
    embedding_key: str = "x"
    embedding_max_lengths: dict = field(default={"esm": 2048, "seqLens": 512})

    @property
    def datasets(self) -> Path:
        return self.outdir / "datasets"

    def get_outputs(self) -> list:
        out = [
            self.outdir / "label_cooccurrence.csv",
            self.outdir / "cooccurrence_stats.yaml",
        ]
        custom_saes: dict = self.saes["custom"]
        for st in SeqTypes:
            if st not in self.embedding_methods or st not in self.fastas:
                continue
            out.extend(
                expand(
                    f"{self.outdir}/training_indices/{{l}}_{st.value}.json",
                    l=[v.value for v in Levels],
                )
            )
            acts_prefix: str = f"{self.datasets}/activations_{st.value}"
            sae_prefix: str = f"{self.outdir}/saes_{st.value}"
            embedding_prefix: str = f"{self.datasets}/embedded_{st.value}"

            for mname, mspec in self.embedding_methods[st].items():
                out.append(f"{embedding_prefix}_tokens/{mname}-0.completed")

                for s in custom_saes.keys():
                    out.append(f"{sae_prefix}_tokens/{mname}-0-{s}.pt")
                    out.append(f"{acts_prefix}_tokens/{mname}-0-{s}")

                for p in mspec.poolings:
                    out.append(f"{embedding_prefix}_seqs/{mname}-{p.value}.completed")
                    for sae in custom_saes:
                        out.append(f"{sae_prefix}_seqs/{mname}-{p.value}-{sae}.pt")
                        out.append(f"{acts_prefix}_seqs/{mname}-{p.value}-{sae}")

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
    return cfg
