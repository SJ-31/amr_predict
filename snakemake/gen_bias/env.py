#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import cattrs
import pandera.polars as pa
import polars as pl
import yaml
from attrs import asdict, define, field, validators
from yte import process_yaml

SCHEMA: pa.DataFrameSchema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, unique=True),
        "perturbed": pa.Column(str, nullable=True),
        "family": pa.Column(str, nullable=True),
        "n": pa.Column(int, nullable=True),
        "taxid": pa.Column(str),
        "biotype": pa.Column(str),
        "has_5p_utr": pa.Column(bool),
        "proportion": pa.Column(float),
        "coding": pa.Column(bool),
        # "conservation": pa.Column(), # TODO: not sure how to do this yet
    }
)


@define
class ModelParams:
    script: (
        str  # the model generation script, either <name>.sh or <name>.py e.g. evo2.py
    )
    env: str  # Conda environment or path to environment yaml
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
        SCHEMA.validate(self.prefix_metadata)
        self.prefixes.extend(self.prefix_metadata["name"].to_list())

    def model_env(self, key: str) -> str:
        return self.models[key].env

    def model_script(self, key: str) -> str:
        """
        Return the `script` field for the entry in self.models
        """
        return self.models[key].script

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
