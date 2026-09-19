#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import cattrs
import pandera.polars as pa
import polars as pl
import yaml
from attrs import asdict, define, field, validators
from yte import process_yaml


def cols_not_all_null(data: pa.PolarsData, a: str, b: str) -> pl.LazyFrame:
    return data.lazyframe.select(pl.col(a).is_not_null() | pl.col(b).is_not_null())


SCHEMA: pa.DataFrameSchema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, unique=True),
        "perturbed": pa.Column(str, nullable=True),
        "file": pa.Column(
            str, nullable=True, checks=pa.Check(lambda x: x.exists(), element_wise=True)
        ),
        "seq": pa.Column(str, nullable=True),
        "family": pa.Column(str, nullable=True),
        "n": pa.Column(int, nullable=True),
        "taxid": pa.Column(str),
        "biotype": pa.Column(str),
        "has_5p_utr": pa.Column(bool),
        "proportion": pa.Column(float),
        "coding": pa.Column(bool),
        "motif": pa.Column(str, nullable=True),
        # "conservation": pa.Column(), # TODO: not sure how to do this yet
    },
    checks=pa.Check(cols_not_all_null, "file", "seq"),
)


@define
class ModelParams:
    script: (
        str  # the model generation script, either <name>.sh or <name>.py e.g. evo2.py
    )
    image: str  # Image file
    kws: dict = field(factory=dict)


@define
class FindMotifs:
    default: Path


@define
class SnakeEnv:
    huggingface: str
    rng: int
    models: dict[str, ModelParams]
    slurm_time_limit: str
    resources: dict = field(validator=validators.instance_of(dict[str, dict[str, str]]))
    n: int
    meta: pl.DataFrame = field(converter=pl.read_csv)
    outdir: Path = field(converter=Path)
    tmp: Path = field(converter=Path)
    prefixes: list[str] = field(init=False, factory=list)
    prefix2file: dict[str, str] = field(init=False, factory=dict)

    def __attrs_post_init__(self):
        SCHEMA.validate(self.meta)
        if not self.tmp.exists():
            self.tmp.mkdir()
        self.prefixes.extend(self.meta["name"].to_list())
        for prefix, file, seq in zip(
            self.meta["name"], self.meta["file"], self.meta["seq"]
        ):
            if not file and seq:
                file = self.tmp / f"{prefix}.fasta"
                file.write_text(f">{prefix}\n{seq}")
            self.prefix2file[prefix] = file

    def model_image(self, key: str) -> str:
        return self.models[key].image

    def model_script(self, key: str) -> str:
        """
        Return the `script` field for the entry in self.models
        """
        return self.models[key].script

    def get_outputs(self) -> dict:
        """Return a dictionary of all workflow outputs, as input to the top-level rule"""
        results = {"generated": [], "metrics": [], "motifs": []}
        for model in self.models:
            for prefix in self.prefixes:
                results["generated"].append(
                    str(self.outdir / "generated" / model / f"{prefix.name}.fasta")
                )
                results["motifs"].append(
                    str(self.outdir / "motifs" / model / f"{prefix.name}.tsv")
                )
        results["metrics"].append("prefix_comparison.csv")
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
