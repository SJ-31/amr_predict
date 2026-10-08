#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import cattrs
import pandera.polars as pa
import polars as pl
import yaml
from attrs import asdict, define, field, validators
from Bio import SeqIO
from snakemake.io import expand
from yte import process_yaml


def cols_not_all_null(data: pa.PolarsData, a: str, b: str) -> pl.LazyFrame:
    return data.lazyframe.select(pl.col(a).is_not_null() | pl.col(b).is_not_null())


def check_fasta(file: str) -> bool:
    try:
        record = SeqIO.read(file, "fasta")
        return len(record) > 0
    except (ValueError, FileNotFoundError):
        return False


BIOTYPES: list[str] = [
    # Protein coding
    "protein_coding",
    "protein_coding_CDS_not_defined",
    "protein_coding_LoF",
    "nonsense_mediated_decay",
    "stop_codon_readthrough",
    "readthrough_transcript",
    # Immunoglobulin / T cell receptor genes
    "IG_C_gene",
    "IG_D_gene",
    "IG_J_gene",
    "IG_V_gene",
    "TR_C_gene",
    "TR_D_gene",
    "TR_J_gene",
    "TR_V_gene",
    # Pseudogenes
    "pseudogene",
    "IG_pseudogene",
    "polymorphic_pseudogene",
    "processed_pseudogene",
    "unprocessed_pseudogene",
    "unitary_pseudogene",
    "transcribed_pseudogene",
    "transcribed_processed_pseudogene",
    "transcribed_unprocessed_pseudogene",
    "transcribed_unitary_pseudogene",
    "translated_pseudogene",
    # Processed transcript / long non-coding
    "processed_transcript",
    "lncRNA",
    "lincRNA",
    "antisense",
    "3prime_overlapping_ncRNA",
    "macro_lncRNA",
    "non_coding",
    "retained_intron",
    "sense_intronic",
    "sense_overlapping",
    # Small non-coding
    "ncRNA",
    "miRNA",
    "misc_RNA",
    "piRNA",
    "rRNA",
    "siRNA",
    "snRNA",
    "snoRNA",
    "tRNA",
    "vault_RNA",
    # Other
    "TEC",
]

SCHEMA: pa.DataFrameSchema = pa.DataFrameSchema(
    {
        "name": pa.Column(
            str,
            unique=True,
            checks=pa.Check(
                lambda x: x.lazyframe.select(~pl.col(x.key).str.contains("-"))
            ),
        ),
        "perturbed": pa.Column(bool, nullable=True),
        "file": pa.Column(
            str,
            nullable=True,
            checks=pa.Check(check_fasta, element_wise=True),
        ),
        "file_full": pa.Column(  # FASTA file containing the original
            # sequence of the prompt
            str,
            checks=pa.Check(check_fasta, element_wise=True),
        ),
        "seq": pa.Column(str, nullable=True),
        "n": pa.Column(int, nullable=True),
        "taxid": pa.Column(str, coerce=True),
        "biotype": pa.Column(str, checks=pa.Check.isin(BIOTYPES)),
        "has_5p_utr": pa.Column(bool),
        "proportion": pa.Column(float),
        "gen_length": pa.Column(int),
        "motif_file": pa.Column(
            str,
            nullable=True,
            checks=pa.Check(lambda x: Path(x).exists(), element_wise=True),
        ),
        "batch_size": pa.Column(
            int, nullable=True, checks=pa.Check.greater_than(0), coerce=True
        ),
        # "tree_file": pa.Column(
        #     str,
        #     checks=pa.Check(lambda x: Path(x).exists(), element_wise=True),
        # ),
    },
    checks=pa.Check(cols_not_all_null, a="file", b="seq"),
    strict="filter",
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
class ParasailParams:
    fn: str = "sw_stats_striped_16"
    gap_extend: int = 1
    gap_open: int = 10
    matrix: str = "blosum62"
    match: int = 1
    mismatch: int = 0


@define
class SnakeEnv:
    huggingface: str
    rng: int
    models: dict[str, ModelParams]
    slurm_time_limit: str
    resources: dict = field(validator=validators.instance_of(dict))
    n: int
    fimo: FindMotifs
    meta: pl.DataFrame = field(
        converter=lambda x: pl.read_csv(
            x, null_values="NA", schema_overrides={"n": pl.Int64}
        )
        if not isinstance(x, pl.DataFrame)
        else x
    )
    outdir: Path = field(converter=Path)
    taxdb: str
    tmp: Path = field(converter=Path)
    singularity: dict = field(factory=dict)
    gen_batch_size: int = 15
    parasail: ParasailParams = field(factory=ParasailParams)
    resource_mappings: dict[str, str] = field(factory=dict)
    prompts: list[str] = field(init=False, factory=list)
    prompt2data: dict[str, dict] = field(init=False, factory=dict)

    def __attrs_post_init__(self):
        self.meta = SCHEMA.validate(self.meta).with_columns(
            pl.col("biotype").is_in(BIOTYPES[:6]).alias("coding")
        )
        if not self.tmp.exists():
            self.tmp.mkdir()
        self.prompts.extend(self.meta["name"].to_list())
        self.prompt2data = self.meta.rows_by_key("name", unique=True, named=True)

    def get_motif_file(self, prompt: str) -> str:
        return (
            self.prompt2data[prompt].get("motif_file", self.fimo.default)
            or self.fimo.default
        )

    def get_resources(self, rule: str) -> dict:
        if mapped := self.resource_mappings.get(rule):
            return self.resources.get(mapped, {})
        return {}

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
            ("generated", ["fasta", "csv"]),
            ("motifs", "tsv"),
            ("taxonomy", "csv"),
            ("nuc_similarity", (("self", "to_prompt"), "csv")),
            ("aa_similarity", (("self", "to_prompt"), "csv")),
            ("protein_descriptors", (("means", "raw"), "csv")),
        ]:
            kws = {"m": self.models.keys(), "p": self.prompts}
            if isinstance(ext, str):
                results[d] = expand(f"{self.outdir}/{d}/{{m}}/{{p}}.{ext}", **kws)
            elif isinstance(ext, list):
                results[d] = expand(
                    f"{self.outdir}/{d}/{{m}}/{{p}}.{{e}}", e=ext, **kws
                )
            else:
                suffixes, ext = ext
                results[d] = expand(
                    f"{self.outdir}/{d}/{{m}}/{{p}}-{{t}}.{ext}", t=suffixes, **kws
                )
        for m in ["prompt_comparison.csv", "physicochemical.csv"]:
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
    print(data)
    return SnakeEnv.new(data)


def test_full() -> SnakeEnv:
    from pyhere import here

    wd = here("snakemake", "gen_bias")
    with open(wd / "env.yaml", "r") as f:
        data = process_yaml(f)
    with open(wd / "all_models.yaml", "r") as f:
        data.update(process_yaml(f))
    print(data)
    return SnakeEnv.new(data)
