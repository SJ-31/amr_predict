#!/usr/bin/env python3

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path

import cattrs
import yaml
from amr_predict.utils import SeqTypes
from attr.validators import instance_of
from attrs import asdict, define, field, validators
from snakemake.io import expand
from yte import process_yaml

Levels = Enum("Levels", (("TOKENS", "tokens"), ("SEQS", "seqs")))


@define
class SnakeEnv:
    huggingface: str
    resources: dict = field(validator=instance_of(dict))

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
