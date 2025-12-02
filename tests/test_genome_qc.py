#!/usr/bin/env ipython

import subprocess as sp
from pathlib import Path
from tempfile import NamedTemporaryFile

import polars as pl
import pytest
import yaml
from loguru import logger
from pyhere import here

logger.add(here("tests", "genome_qc.log"))

GENOMES: Path = here("data", "remote", "genomes")
REF = GENOMES.joinpath("reference")
OUT: Path = here("results", "tests")

A_BAUMANNI = [
    str(GENOMES.joinpath("jia", f))
    for f in [
        "SAMN29490345.fasta",
        "SAMN29490346.fasta",
        "SAMN29490347.fasta",
        "SAMN29490348.fasta",
        "SAMN29490349.fasta",
        "SAMN29490350.fasta",
    ]
]
E_COLI = [
    str(GENOMES.joinpath("moradigaravand", f))
    for f in [
        "SAMN29490345.fasta",
        "SAMN29490346.fasta",
        "SAMN29490347.fasta",
        "SAMN29490348.fasta",
        "SAMN29490349.fasta",
        "SAMN29490350.fasta",
        "ERR434259.fasta",
        "ERR434260.fasta",
        "ERR434261.fasta",
        "ERR434262.fasta",
        "ERR434263.fasta",
        "ERR434264.fasta",
        "ERR434265.fasta",
        "ERR434266.fasta",
    ]
]

tmp = {
    "a.fa": 1001,
    "b.fa": 1001,  # same taxid as a.fa
    "c.fa": 1002,
    "d.fa": 1003,
    "e.fa": 1003,  # same taxid as d.fa
    "f.fa": 1004,
}
query2taxid = pl.DataFrame({"query": tmp.keys(), "taxid": tmp.values()})
tmp = {
    1001: ["a.fa", "b.fa"],
    1002: ["c.fa"],
    1003: ["d.fa", "e.fa", "f.fa"],
}
taxid2ref = pl.DataFrame({"taxid": tmp.keys(), "reference": tmp.values()})

q2ref = pl.DataFrame(
    {
        "query": ["q1", "q1", "q1", "q2", "q2", "q3", "q3", "q3"],
        "reference": ["a1", "a2", "a3", "a4", "a5", "a1", "a2", "a3"],
    }
)


def get_refs(*names):
    return [REF.joinpath(n) for n in names]


@pytest.fixture
def make_file_spec():
    tmp = NamedTemporaryFile(suffix=".txt")

    def f(*args, **kwargs):
        first, second = kwargs.get("first"), kwargs.get("second")
        if first is None:
            to_write = [it for lst in args for it in lst]
            Path(tmp.name).write_text("\n".join(to_write))
        else:
            first, second = kwargs["first"], kwargs["second"]
            f, s = args
            pl.DataFrame({first: f, second: s})
        return tmp.name

    yield f
    tmp.close()


@pytest.fixture
def make_conf():
    tmp = NamedTemporaryFile(suffix=".yaml")

    def f(config):
        with open(tmp.name, "w") as f:
            yaml.safe_dump(config, f)
        return tmp.name

    yield f
    tmp.close()


def check_py_script(command):
    proc = sp.run(f"python {command}", shell=True, stdout=True)
    proc.check_returncode()


def test_fcs(make_file_spec, make_conf):
    inputs = make_file_spec(A_BAUMANNI)
    config = make_conf(
        {
            "run": {
                "fcs": {
                    "fasta-input": inputs,
                    "image": "~/tools/fcs-adaptor.sif",
                    "taxonomy": "prok",
                    "container-engine": "singularity",
                }
            }
        }
    )
    logger.info(Path(config).exists())
    logger.info(Path(inputs).exists())
    logger.info(Path(inputs).read_text())
    out = OUT.joinpath("fcs")
    command = f"{here("scripts", "genome_qc.py")} {config} --run fcs --outdir {out} --output {OUT.joinpath("fcs.tsv")}"
    check_py_script(command)


def test_fastani_q2t(make_file_spec, make_conf):
    qt2 = make_file_spec(
        [f for group in [E_COLI, A_BAUMANNI] for f in group],
        ["1"] * len(E_COLI) + ["2"] * len(A_BAUMANNI),
        first="query",
        second="taxid",
    )
    logger.info(qt2)
    t2r = make_conf(
        {
            "1": get_refs(
                "GCF_000005845.2_ASM584v2_genomic.fna",
                "GCF_000008865.2_ASM886v2_genomic.fna",
            ),
            "2": get_refs(
                "GCA_009759685.1_ASM975968v1_genomic.fna",
                "GCA_014672755.1_ASM1467275v1_genomic.fna",
                "GCA_020911985.1_ASM2091198v1_genomic.fna",
            ),
        }
    )
    config = make_conf(
        {
            "run": {
                "fastani": {
                    "mapping": {
                        "query2taxid": qt2,
                        "taxid2reference": t2r,
                    }
                }
            }
        }
    )
    out = OUT.joinpath("fastani")
    command = f"{here("scripts", "genome_qc.py")} {config} --run fastani --outdir {out} --output {OUT.joinpath("fastani.tsv")}"
    check_py_script(command)


def test_fastani_q2r(make_file_spec, make_conf):
    q2r = make_file_spec(
        E_COLI,
        get_refs("GCF_000005845.2_ASM584v2_genomic.fna") * len(E_COLI),
        first="query",
        second="reference",
    )
    config = make_conf(
        {
            "run": {
                "fastani": {
                    "mapping": {
                        "query2reference": q2r,
                    }
                }
            }
        }
    )
    out = OUT.joinpath("fastani_q2r")
    command = f"{here("scripts", "genome_qc.py")} {config} --run fastani --outdir {out} --output {OUT.joinpath("fastani_q2r.tsv")}"
    check_py_script(command)


# def test_filtering():
#     config = {"kraken2": [], "kraken2_expected_taxids": []}
