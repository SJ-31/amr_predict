#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import cattrs
import pandera.polars as pa
import polars as pl
import yaml
from attrs import asdict, define, field, validators
from snakemake.io import expand
from yte import process_yaml


def cols_not_all_null(data: pa.PolarsData, a: str, b: str) -> pl.LazyFrame:
    return data.lazyframe.select(pl.col(a).is_not_null() | pl.col(b).is_not_null())


SCHEMA: pa.DataFrameSchema = pa.DataFrameSchema(
    {
        "name": pa.Column(str, unique=True),
        "perturbed": pa.Column(bool, nullable=True),
        "file": pa.Column(
            str,
            nullable=True,
            checks=pa.Check(lambda x: Path(x).exists(), element_wise=True),
        ),
        "file_full": pa.Column(  # FASTA file containing the original
            # sequence of the prefix
            str,
            checks=pa.Check(lambda x: Path(x).exists(), element_wise=True),
        ),
        "seq": pa.Column(str, nullable=True),
        "family": pa.Column(str, nullable=True),
        "n": pa.Column(int, nullable=True),
        "taxid": pa.Column(str, coerce=True),
        "biotype": pa.Column(str),
        "has_5p_utr": pa.Column(bool),
        "proportion": pa.Column(float),
        "coding": pa.Column(bool),
        "motif_file": pa.Column(
            str,
            nullable=True,
            checks=pa.Check(lambda x: Path(x).exists(), element_wise=True),
        ),
        # "conservation": pa.Column(), # TODO: not sure how to do this yet
    },
    checks=pa.Check(cols_not_all_null, a="file", b="seq"),
)


@define
class ModelParams:
    script: (
        str  # the model generation script, either <name>.sh or <name>.py e.g. evo2.py
    )
    kws: dict = field(factory=dict)
    resources: str | None = None


@define
class FindMotifs:
    default: str
    thresh: float = 1e-5


@define
class SnakeEnv:
    huggingface: str
    rng: int
    models: dict[str, ModelParams]
    slurm_time_limit: str
    resources: dict = field(validator=validators.instance_of(dict))
    n: int
    fimo: FindMotifs
    meta: pl.DataFrame = field(converter=lambda x: pl.read_csv(x, null_values="NA"))
    outdir: Path = field(converter=Path)
    taxdb: str
    tmp: Path = field(converter=Path)
    singularity: dict = field(factory=dict)
    prefixes: list[str] = field(init=False, factory=list)
    prefix2data: dict[str, dict] = field(init=False, factory=dict)
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
        self.prefix2data = self.meta.rows_by_key("name", unique=True, named=True)

    def get_prefix_file(self, prefix: str, full: bool = False):
        if not full:
            return self.prefix2file[prefix]
        return self.prefix2data[prefix]["file_full"]

    def get_motif_file(self, prefix: str) -> str:
        return (
            self.prefix2data[prefix].get("motif_file", self.fimo.default)
            or self.fimo.default
        )

    def model_image(self, key: str) -> str | None:
        val = self.singularity.get("generate", {})
        if isinstance(val, dict):
            return val.get(key)
        return val

    def model_res(self, key: str) -> dict:
        res = self.models[key].resources
        return self.resources[res] if res else {}

    def model_kws(self, key: str) -> str:
        return " ".join(
            [f"--{k} {v}" if v else f"--{k}" for k, v in self.models[key].kws.items()]
        )

    def model_script(self, key: str) -> str:
        """
        Return the `script` field for the entry in self.models
        """
        return self.models[key].script

    def get_outputs(self) -> dict:
        """Return a dictionary of all workflow outputs, as input to the top-level rule"""
        results = {"metrics": []}
        for d, ext in [
            ("generated", "fasta"),
            ("motifs", "tsv"),
            ("taxonomy", "csv"),
        ]:
            results[d] = expand(
                f"{self.outdir}/{d}/{{m}}/{{p}}.{ext}",
                m=self.models.keys(),
                p=self.prefixes,
            )
        for m in ["prefix_comparison.csv", "physicochemical.csv"]:
            results["metrics"].append(f"{self.outdir}/{m}")
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


def test_env() -> SnakeEnv:
    from pyhere import here

    wd = here("snakemake", "gen_bias")
    with open(wd / "env.yaml", "r") as f:
        data = process_yaml(f)
    with open(wd / "test_env.yaml", "r") as f:
        data.update(process_yaml(f))
    return SnakeEnv.new(data)
