#!/usr/bin/env python

import operator
import re
import subprocess as sp
from functools import reduce
from pathlib import Path
from typing import get_args

import polars as pl
import yaml

AVAILABLE_QC = ("kraken2", "quast", "fastani")
AVAILABLE_WRAPPERS = ("fcs", "fastani")

SEQ_RE: str = ".?(fasta|fna|faa|fa)(.gz)?"

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
    "kraken2": {"min_percent_expected": 2, "max_percent_other": 0.3},
}


def get_spec(name: str, config: dict) -> dict:
    """Return specification for filter method NAME after removing mappings with nulls"""
    conf = config.get(name, DEFAULTS.get(name))
    valid = {k: v for k, v in conf.items() if v is not None}
    return valid


def fastani_wrapper(outdir: Path, config: dict) -> pl.DataFrame:
    if "mapping" not in config:
        raise ValueError("`mapping` must be given for fastANI")
    maps: dict[str, pl.DataFrame] = {}
    command = "fastANI "
    for k, v in config.items():
        if k != "mapping":
            command += f" --{k} {v} "

    def run_fastani(command, group: pl.DataFrame, group_name) -> pl.DataFrame:
        rl = outdir.joinpath(f"{group_name}_refs.txt")
        ql = outdir.joinpath(f"{group_name}_queries.txt")
        outfile = outdir.joinpath(f"{group_name}_result.tsv")
        if not outfile.exists():
            rl.write_text("\n".join(group["reference"].first()))
            ql.write_text("\n".join(group["query"]))
            for path, flag in zip([ql, rl], ["--ql", "--rl"]):
                if path is not None:
                    command += f" {flag} {path} "
            command += f" --output {outfile}"
            proc = sp.run(command, shell=True)
            proc.check_returncode()
        return pl.read_csv(
            outfile,
            separator="\t",
            new_columns=(
                "query",
                "reference",
                "ANI",
                "bidirectional_fragments",
                "total_query_fragments",
            ),
            has_header=False,
        )

    dfs = []
    for path, headers in zip(
        ["query2reference", "query2taxid", "taxid2reference"],
        (("query", "reference"), ("query", "taxid"), None),
    ):
        if p := config["mapping"].get(path):
            if headers is not None:
                maps[path] = pl.read_csv(
                    p, separator="\t", new_columns=headers, has_header=False
                ).cast(pl.String)
            else:
                with open(p, "r") as f:
                    tmp = yaml.safe_load(f)
                    maps[path] = pl.DataFrame(
                        {"taxid": tmp.keys(), "reference": tmp.values()}
                    )
    q2t, t2r = maps.get("query2taxid"), maps.get("taxid2reference")
    if (q2r := maps.get("query2reference")) is None and (q2t is None or t2r is None):
        raise ValueError(
            "Either `query2reference` or both `query2taxid` and `taxid2reference` must be given"
        )
    if not outdir.exists():
        outdir.mkdir()
    if q2t is not None and t2r is not None:
        merged = q2t.join(t2r, on="taxid")
        for taxid, group in merged.group_by("taxid"):
            # "many-to-one" mode: many queries, one reference genome
            df = run_fastani(command=command, group_name=taxid[0], group=group)
            dfs.append(df)
    if q2r is not None:
        for i, (_, group) in enumerate(
            q2r.group_by("query").agg("reference").group_by("reference")
        ):
            # "one-to-many" mode: one query, multiple references
            df = run_fastani(command=command, group_name=f"group_{i}", group=group)
            dfs.append(df)
    return pl.concat(dfs).with_columns(
        pl.col("query").str.replace(SEQ_RE, "").str.replace(".*/", "").alias("sample")
    )


def fcs_wrapper(outdir: Path, config: dict) -> pl.DataFrame:
    with open(config["fasta-input"], "r") as f:
        samples = f.read().splitlines()
    dfs = []
    image = config.get("image")
    engine = config.get("container-engine", "singularity")
    taxonomy = config.get("taxonomy", "prok")
    taxonomy_flag = "--prok" if taxonomy == "prok" else "--euk"
    if not image and engine == "singularity":
        raise ValueError("`image` must be defined if using singularity")
    command = f" --container-engine {engine} --image {image} {taxonomy_flag}"
    if not outdir.exists():
        outdir.mkdir()
    for s in samples:
        sample = re.sub(SEQ_RE, "", Path(s).stem)
        cur_out = outdir.joinpath(sample)
        report = cur_out.joinpath("fcs_adaptor_report.txt")
        if not report.exists():
            cur_out.mkdir()
            cur_command = f"--fasta-input {s} --output-dir {cur_out} {command}"
            proc = sp.run(
                f"run_fcsadaptor.sh {cur_command}", shell=True, capture_output=True
            )
            print(proc.stdout.decode())
            try:
                proc.check_returncode()
            except sp.CalledProcessError as e:
                raise e
        df = pl.read_csv(report, separator="\t").with_columns(
            pl.lit(sample).alias("sample")
        )
        if df.shape[0] > 1:
            dfs.append(df)
    if dfs:
        return pl.concat(dfs)
    return pl.DataFrame(
        {
            "#accession": [],
            "length": [],
            "action": [],
            "range": [],
            "name": [],
            "sample": [],
        }
    )


def filter_w_operators(df: pl.DataFrame, spec: list, all: bool = False) -> pl.DataFrame:
    exprs = []
    for col, op, value in spec:
        expr = (OMAP[op](df[col], value)).alias(col)
        exprs.append(expr)
    if all:
        return df.filter(pl.all_horizontal(*exprs))
    return df.filter(pl.any_horizontal(*exprs))


def filter_fastani(spec: dict, file: Path) -> list:
    df = pl.read_csv(file, separator="\t")
    exprs = []
    for k, v in spec.items():
        operator, value = v
        op = OMAP[operator]
        if k == "ani_avg":
            exprs.append(op(pl.col("ANI").mean(), value).alias(k))
        elif k == "ani_all":
            exprs.append(op(pl.col("ANI"), value).all().alias(k))
        elif k == "ani_any":
            exprs.append(op(pl.col("ANI"), value).any().alias(k))
    agg = df.group_by("sample").agg(*exprs)
    filtered = agg.filter(pl.all_horizontal(*spec.keys()))
    return list(filtered["sample"])


def filter_quast(spec: dict, file: Path) -> list:
    df = pl.read_csv(file, separator="\t")
    df = df.rename({"Assembly": "sample"}).with_columns(
        pl.col("sample").str.strip_suffix(".scaffolds")
    )
    filtered = filter_w_operators(df, spec=spec["filters"], all=True)
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
            schema={
                "p_reads_covered": pl.Float64,
                "n_reads_covered": pl.Int64,
                "n_reads_direct": pl.Int64,
                "rank": pl.String,
                "taxid": pl.Int64,
                "name": pl.String,
            },
            has_header=False,
        )
        expected_tax = sample2exptax.get(sample)
        if not expected_tax:
            raise ValueError(
                f"The expected taxon for sample {sample} (file {file}) is not present in the sample->taxid mapping"
            )
        at_expected = df.filter(pl.col("taxid") == expected_tax)
        if min_percent_exp is not None:
            if at_expected["p_reads_covered"].item() < min_percent_exp:
                return
        if max_percent_other is not None:
            rank = at_expected["rank"].item()
            others = df.filter(
                (pl.col("rank") == rank) & (pl.col("taxid") != expected_tax)
            )
            # TODO: maybe want to add a version that uses n_reads
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
        "-r",
        "--run",
        default=None,
        help=f"Run a wrapper for a QC program. Currently supported: {get_args(AVAILABLE_WRAPPERS)}",
        action="store",
    )
    parser.add_argument(
        "config",
        help="Config file specifying QC thresholds and file paths",
        action="store",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output",
        default=None,
    )
    parser.add_argument(
        "-d",
        "--outdir",
        default=None,
        help="A directory to save sample-specific output for the QC wrappers. Required for `--run`",
        action="store",
    )
    args = vars(parser.parse_args())  # convert to dict
    return args


if __name__ == "__main__":
    args = parse_args()
    with open(args["config"], "r") as path:
        conf: dict = yaml.safe_load(path)

    if (to_run := args.get("run")) in AVAILABLE_WRAPPERS:
        run_conf = conf.get("run")
        if not run_conf:
            raise ValueError("No run configuration provided")
        if not run_conf.get(to_run):
            raise ValueError(f"Configuration for `{to_run}` not found in config")
        if to_run == "fcs":
            summary = fcs_wrapper(outdir=Path(args["outdir"]), config=run_conf[to_run])
        elif to_run == "fastani":
            summary = fastani_wrapper(
                outdir=Path(args["outdir"]), config=run_conf[to_run]
            )
        summary.write_csv(args["output"], separator="\t")
        exit(0)
    path_map: dict = conf.get("paths")
    if not path_map:
        raise ValueError("No paths to qc files provided in config")
    passing: dict = {}
    tax_mapping: dict | None = None
    for qc in AVAILABLE_QC:
        paths = path_map.get(qc)
        if not paths:
            continue
        passing[qc] = []
        if qc == "kraken2":
            expected_tax_file = path_map.get("kraken2_expected_taxids")
            if not expected_tax_file:
                raise ValueError(
                    "`kraken2_expected_taxids` field must not be empty to filter with kraken2"
                )
            tax_df = pl.read_csv(
                expected_tax_file,
                separator="\t",
                new_columns=["sample", "taxid"],
                has_header=False,
            )
            tax_mapping = dict(zip(tax_df["sample"], tax_df["taxid"]))
        spec = get_spec(qc, config=conf)
        if not isinstance(paths, list):
            paths = [paths]
        for path in (Path(f) for f in paths):
            if qc == "kraken2":
                passed = filter_kraken2(tax_mapping, spec, path=path)
            elif qc == "quast":
                passed = filter_quast(spec, path)
            elif qc == "fastani":
                passed = filter_fastani(spec, path)
            else:
                raise ValueError("Not qc")
            passing[qc].extend(passed)
    intersection = list(reduce(lambda x, y: x & y, [set(v) for v in passing.values()]))
    if out := args["output"]:
        with open(out, "w") as o:
            o.write("\n".join(intersection))
    else:
        print("\n".join(intersection))
