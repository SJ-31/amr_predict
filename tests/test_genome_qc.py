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

NF_OUT: Path = here("data", "remote", "output")
GENOMES: Path = here("data", "remote", "genomes")
REF = GENOMES.joinpath("reference")
OUT: Path = here("results", "tests")
SCRIPT = here("scripts", "genome_qc.py")

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
    return [str(REF.joinpath(n)) for n in names]


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
            pl.DataFrame({first: f, second: s}).write_csv(tmp.name, separator="\t")
        return tmp.name

    yield f
    tmp.close()


@pytest.fixture
def make_conf(tmp_path):
    def f(name, config):
        file = tmp_path / f"{name}.yaml"
        with open(file, "w") as f:
            yaml.safe_dump(config, f)
        return str(file)

    yield f


def check_py_script(command):
    proc = sp.run(f"python {command}", shell=True, stdout=True)
    proc.check_returncode()


def test_fcs(make_file_spec, make_conf):
    inputs = make_file_spec(A_BAUMANNI)
    config = make_conf(
        "config",
        {
            "run": {
                "fcs": {
                    "fasta-input": inputs,
                    "image": "~/tools/fcs-adaptor.sif",
                    "taxonomy": "prok",
                    "container-engine": "singularity",
                }
            }
        },
    )
    logger.info(Path(config).exists())
    logger.info(Path(inputs).exists())
    out = OUT.joinpath("fcs")
    command = (
        f"{SCRIPT} {config} --run fcs --outdir {out} --output {OUT.joinpath("fcs.tsv")}"
    )
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
        "map",
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
        },
    )
    logger.info(t2r)
    config = make_conf(
        "config",
        {
            "run": {
                "fastani": {
                    "mapping": {
                        "query2taxid": qt2,
                        "taxid2reference": t2r,
                    }
                }
            }
        },
    )
    logger.info(config)
    out = OUT.joinpath("fastani")
    command = f"{SCRIPT} {config} --run fastani --outdir {out} --output {OUT.joinpath("fastani.tsv")}"
    check_py_script(command)


def test_fastani_q2r(make_file_spec, make_conf):
    q2r = make_file_spec(
        E_COLI,
        get_refs("GCF_000005845.2_ASM584v2_genomic.fna") * len(E_COLI),
        first="query",
        second="reference",
    )
    config = make_conf(
        "config",
        {
            "run": {
                "fastani": {
                    "mapping": {
                        "query2reference": q2r,
                    }
                }
            }
        },
    )
    out = OUT.joinpath("fastani_q2r")
    command = f"{SCRIPT} {config} --run fastani --outdir {out} --output {OUT.joinpath("fastani_q2r.tsv")}"
    check_py_script(command)


def test_filter_kraken(make_file_spec, make_conf):
    exp_tax = make_file_spec(
        [
            "ERR434259",
            "ERR434260",
            "ERR434261",
            "ERR434262",
            "ERR434263",
        ],
        [562] * 5,
        first="sample",
        second="taxid",
    )
    paths = {
        "kraken2": [
            str(here(NF_OUT, "moradigaravand_2025-10-06/Kraken2", f))
            for f in [
                "ERR434259.kraken2.report.txt",
                "ERR434260.kraken2.report.txt",
                "ERR434261.kraken2.report.txt",
                "ERR434262.kraken2.report.txt",
                "ERR434263.kraken2.report.txt",
            ]
        ],
        "kraken2_expected_taxids": exp_tax,
    }
    filters = {"min_percent_expected": 20, "max_percent_other": 0.1}
    config = make_conf("config", {"paths": paths, "kraken2": filters})
    o = OUT.joinpath("qc_kraken2_filter.txt")
    command = f"{SCRIPT} {config} --output {o}"
    check_py_script(command)
    passing = {"ERR434260", "ERR434261"}
    received = set(o.read_text().splitlines())
    assert passing == received


def test_file_quast(make_conf):
    paths = {
        "quast": str(here(NF_OUT, "moradigaravand_2025-10-06/QUAST/report/report.tsv")),
    }
    filters = {"# contigs": ["<", 100], "Total length": [">", 5000000]}
    config = make_conf("config", {"paths": paths, "quast": filters})
    out = OUT.joinpath("qc_quast_filter.txt")
    command = f"{SCRIPT} {config} --output {out}"
    check_py_script(command)
    received = set(out.read_text().splitlines())
    assert "ERR434265.scaffolds" in received
    assert "ERR434259.scaffolds" not in received


def test_filter_fani(make_conf):
    config = make_conf(
        "config",
        {
            "paths": {"fastani": str(OUT.joinpath("fastani.tsv"))},
            "fastani": {"ani_any": [">", 98], "ani_avg": [">", 97.6]},
        },
    )
    out = OUT.joinpath("fastani_filtered.txt")
    command = f"{SCRIPT} {config} --output {out}"
    check_py_script(command)
    received = set(out.read_text().splitlines())
    assert received == {"ERR434263", "SAMN29490346", "ERR434262"}
