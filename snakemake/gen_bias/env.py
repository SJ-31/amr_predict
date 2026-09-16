#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import cattrs
import polars as pl
import yaml
from attrs import asdict, define, field, validators
from yte import process_yaml


@define
class ModelParams:
    name: str
    kws: dict = field(factory=dict)


@define
class SnakeEnv:
    huggingface: str
    rng: int
    outdir: Path = field(converter=Path)
    prefix_metadata: pl.DataFrame = field(converter=pl.read_csv)
    models: dict[str, ModelParams]
    slurm_time_limit: str
    resources: dict = field(validator=validators.instance_of(dict[str, dict[str, str]]))
    n: int
    prefixes: list[str] = field(factory=list)

    def __attrs_post_init__(self):
        self.prefixes.extend()

    def get_outputs(self) -> dict:
        """Return a dictionary of all workflow outputs, as input to the top-level rule"""
        results = {"generated": [], "metrics": []}
        gen_dir = self.outdir / "generated"
        for model in self.models:
            for prefix in self.prefixes:
                results["generated"].append(str(gen_dir / model / prefix.name))
        results["metrics"].append("prefix_comparison.csv")
        results["metrics"].append("motifs_domains.csv")
        results["metrics"].append("physicochemical.csv")
        return results

    @classmethod
    def new(cls, data: str | dict, with_yte: bool = True) -> SnakeEnv:
        if isinstance(data, str):
            assert Path(data).exists() and data.endswith(
                ".yaml"
            ), "Must pass a yaml file"
            with open(data, "r") as f:
                data = process_yaml(f) if with_yte else yaml.safe_load(f)
                return cattrs.structure(data, SnakeEnv)
        return cattrs.structure(data, SnakeEnv)

    def to_dict(self) -> dict:
        return asdict(self)
