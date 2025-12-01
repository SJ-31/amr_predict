#!/usr/bin/env ipython

# TODO: you can use the methods of https://www.nature.com/articles/s41598-025-24333-9#Sec2 for inspiration
# And also review QC metrics used by the ncbi
# Quast from bacass will provide N50, genome fraction, number of contigs etc.
# Can do ANI with https://github.com/ParBLiSS/FastANI
# TODO: will also need to filter samples by QC metrics after you've downloaded and assembled them.
# Mainly check assembly metrics cause the submitters will hopefully have handled raw reads
# TODO: when downloading, first check if a fasta assembly is available on the ncbi
# TODO: interpret output of quast and of kraken2. Cli interface has option to do ANI
import operator
from pathlib import Path

import polars as pl
import yaml

OMAP: dict = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}
DEFAULTS: dict = {
    "quast": {
        "# contigs (>= 0 bp)": None,
        "# contigs (>= 1000 bp)": None,
        "# contigs (>= 5000 bp)": None,
        "# contigs (>= 10000 bp)": None,
        "# contigs (>= 25000 bp)": None,
        "# contigs (>= 50000 bp)": None,
        "Total length (>= 0 bp)": None,
        "Total length (>= 1000 bp)": None,
        "Total length (>= 5000 bp)": None,
        "Total length (>= 10000 bp)": None,
        "Total length (>= 25000 bp)": None,
        "Total length (>= 50000 bp)": None,
        "# contigs": None,
        "Largest contig": None,
        "Total length": None,
        "GC (%)": None,
        "N50": None,
        "N90": None,
        "auN": None,
        "L50": None,
        "L90": None,
        "# N's per 100 kbp": None,
    },
    "kraken2": {"min_percent_expected": 0.5, "max_percent_other": 0.2},
}


def get_spec(name: str, config: dict) -> dict:
    conf = config.get(name, DEFAULTS.get(name))
    valid = {k: v for k, v in conf.items() if v is not None}
    return valid


def filter_w_operators(
    df: pl.DataFrame, spec: dict, all: bool = False, default_direction: str = ">"
) -> pl.DataFrame:
    exprs = []
    for k, v in spec.items():
        if isinstance(v, list):
            direction, value = v
            expr = (OMAP[direction](df[k], value)).alias(k)
        else:
            expr = (OMAP[default_direction](df[k], value)).alias(k)
        exprs.append(expr)
    if all:
        return df.filter(pl.all_horizontal(*exprs))
    return df.filter(pl.any_horizontal(*exprs))


def filter_quast(spec: dict, file: Path) -> list:
    df = pl.read_csv(file, separator="\t")
    df = df.transpose(
        column_names="Assembly", include_header=True, header_name="sample"
    )
    filtered = filter_w_operators(df, spec=spec, all=True, default_direction=">")
    return list(filtered["sample"])


def filter_kraken2(sample2exptax: dict, spec: dict, path: Path) -> list:
    min_percent_exp = spec.get("min_percent_expected")
    max_percent_other = spec.get("max_percent_other")

    passed = []

    def filter_one(sample, file: Path):
        df = pl.read_csv(
            file,
            separator="\t",
            new_columns=[
                "p_reads_covered",
                "n_reads_covered",
                "n_reads_direct",
                "rank",
                "taxid",
                "name",
            ],
        )
        # TODO: maybe want to add a version that uses n_reads
        expected_tax = sample2exptax.get(sample)
        if not expected_tax:
            raise ValueError(
                f"The expected taxon for sample {sample} (file {file}) is not present in the sample->taxid mapping"
            )
        at_expected = df.filter(pl.col("taxid") == expected_tax)
        if at_expected["p_reads_covered"].item() < min_percent_exp:
            return
        rank = at_expected["rank"].item()
        others = df.filter(pl.col("rank") == rank & pl.col("taxid") != expected_tax)
        if any(others["p_reads_covered"] > max_percent_other):
            return
        passed.append(sample)

    if path.is_dir():
        for file in path.glob("*kraken2.report.txt"):
            sample = file.stem.removesuffix(".kraken2.report")
            filter_one(sample, file)
    else:
        sample = path.stem.removesuffix(".kraken2.report")
        filter_one(sample, path)
    return passed


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        required=True,
        help="Config file specifying QC thresholds and file paths",
        action="store",
    )
    parser.add_argument("-o", "--output", help="Path to output file", default=None)
    args = vars(parser.parse_args())  # convert to dict
    return args


test = "/home/shannc/Bio_SDD/stem_synology/chula_mount/shannc/repos/amr_predict/output/moradigaravand_2025-10-06/QUAST/report/report.tsv"

if __name__ == "__main__":
    args = parse_args()
    with open(args["config"], "r") as f:
        conf: dict = yaml.safe_load(f)
    paths = conf.get("paths")
    if not paths:
        raise ValueError("No paths to qc files provided in config")
    # if kraken2 := paths.get("kraken2"):
    # if isinstance(kraken2, list):
    #     for
    # else:
