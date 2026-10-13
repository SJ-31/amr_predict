#!/usr/bin/env ipython

import re

import polars as pl
from Bio import Phylo
from pyhere import here

workdir = here("data", "remote", "ref_trees", "LTP")


mapping = {
    r[1]: r[0]
    for r in pl.read_csv(
        workdir / "acc2name.tsv",
        separator="\t",
        new_columns=["name", "acc"],
        has_header=False,
    ).iter_rows()
}

name2lineage = {
    r[0]: r[1]
    for r in pl.read_csv(workdir / "name2lineage.tsv", separator="\t").iter_rows()
}

new_mapping = {}
dupes = {}

tree = Phylo.read(workdir / "LTPlus_10_2026_clean.nwk", "newick")
for tip in tree.get_terminals():
    acc = re.findall(".*\((.*)\)", tip.name)[0]
    count = dupes.get(acc, 0) + 1
    dupes[acc] = count
    name = mapping[acc]
    suffix = "" if count == 1 else f"_{count}"
    new_mapping[f"{name}{suffix}"] = name2lineage[name]
    tip.name = f"{name}{suffix}"
Phylo.write(tree, workdir / "LTPlus_10_2026_clean_dedup.nwk", "newick")

pl.DataFrame({"name": new_mapping.keys(), "lineage": new_mapping.values()}).write_csv(
    workdir / "name2lineage_dedup.tsv", separator="\t", include_header=False
)
