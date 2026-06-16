#!/usr/bin/env ipython

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import networkx as nx
import numpy as np
import polars as pl
import yaml
from amr_predict.utils import read_tabular
from attrs import define, field
from polars import meta
from pyhere import here
from yte import process_yaml

WD = here("data", "meta", "GO")
WD.mkdir(exist_ok=True)
DF = pl.read_csv(WD / "go_attrs.csv", null_values="NA")
METADATA_FILE = here(
    "data", "remote", "datasets", "2026-04-21_uniprot_swissprot_goa.tsv"
)
G = nx.read_gml(WD / "go.gml", label="name")


def relation_filter(x, y, key, allowed):
    edge = G[x][y][key]
    return edge["relation"] in allowed


GS = nx.subgraph_view(
    G, filter_edge=lambda x, y, z: relation_filter(x, y, z, {"is_a", "part_of"})
)

# TODO: use the blacklist to get rid of highly specific terms, then
# go through the remaining list manually to keep what you think
# is relevant

blacklist = [
    # "nChildrenIsA",
    # "nChildrenPartOf",
    # "nChildrenRegulates",
    "nChildrenHasPart",
    "nChildrenNegativelyRegulates",
    "nChildrenPositivelyRegulates",
    # "nChildrenOccursIn",
    "nChildrenHappensDuring",
    "nChildrenEndsDuring",
]


def get_groups() -> dict:
    groups: dict = {}
    for ns in ["BP", "CC", "MF"]:
        group_file = WD / f"{ns}_groups.yaml"
        if group_file.exists():
            with open(group_file, "r") as f:
                groups[ns] = yaml.safe_load(f)
    return groups


def update_candidate_files():
    groups = get_groups()
    in_group = {
        ns: [v for _, vals in g.items() for v in vals] for ns, g in groups.items()
    }

    def has_paths_with_prev(x, ns) -> bool:
        for node in in_group.get(ns, []):
            # Source must be the potential child node in this format
            if nx.has_path(GS, source=x, target=node):
                return True
        return False

    dist_thresholds = {"CC": 5, "BP": 4, "MF": 5}

    dfs = {
        ns: DF.filter(
            (~pl.any_horizontal([pl.col(b) > 0 for b in blacklist]))
            & (pl.col("maxDistanceToNs") <= dist_thresholds.get(ns, 4))
            & (pl.col("namespace") == ns)
            & (~pl.col("subset").str.contains("gocheck_do_not_annotate"))
            # & (pl.col("subset").is_not_null())
            & (~pl.col("name").is_in(in_group.get(ns, [])))
        )
        for ns in ["BP", "CC", "MF"]
    }
    dfs = {
        k: df.filter(
            ~pl.col("name").map_elements(
                lambda x: has_paths_with_prev(x, ns=k), return_dtype=pl.Boolean
            )
        )
        for k, df in dfs.items()
    }
    for k, v in dfs.items():
        v.write_csv(WD / f"{k}_list.csv")


@define
class GoGroup:
    """
    Class for labelling sets of GO terms to GO groups

    Parameters
    ----------
    cache : dict[tuple[str, str], bool]
        Dictionary caching which GG a term should be assigned to
    """

    G: nx.DiGraph
    ns: Literal["BP", "CC", "MF"]
    groups: dict[str, set]
    cache: dict[str, str] = field(init=False, factory=dict)

    def label(self, terms: Sequence[str]) -> str:
        """
        Label a set of GO terms `terms` with a GO group (GG)

        Filter out the sets of GO terms by their namespace first

        Notes
        -----
        - For each term in a sequence, assign it to a GG (see rules below)
        - The GG with the most terms is the winner. Break ties randomly

        Term assignment rules
        1. If the term is an exact match for a GG member, assign it to that GG
        2. If the term is a child of only one GG member, assign it to that GG
        3. If the term is a child of members of multiple different GGs,
            assign it to the GG whose parent is nearest to the term
            EX: GG1 has "response to topologically incorrect protein",
                GG2 has "response to stress", and the query is
                "cytoplasm protein quality control by the
                    ubiquitin-proteasome system"
                Both terms are parents of the query, but GG1 is more specific
                (shorter distance) so the query goes to GG1
        Returns
        -------
        A GO group name, a key of "groups"
        """
        terms = filter(
            lambda x: x in self.G.nodes and self.G.nodes[x]["namespace"] == self.ns,
            terms,
        )
        group_tracker: dict[str, list] = defaultdict(list)
        for term in terms:
            if cached := self.cache.get(term):
                group_tracker[cached].append(term)
                continue
            path_len_tracker = {}
            for cand, members in self.groups.items():
                if term in members:
                    self.cache[term] = cand
                    group_tracker[cand].append(term)
                    break
                for m in members:
                    if nx.has_path(G, source=term, target=m):
                        dist = nx.shortest_path_length(G, source=term, target=m)
                        path_len_tracker[cand] = min(
                            path_len_tracker.get(cand, np.inf), dist
                        )
            if path_len_tracker:
                selected_group = min(path_len_tracker.items(), key=lambda x: x[1])[0]
                group_tracker[selected_group].append(term)
        if group_tracker:
            chosen = max(group_tracker.items(), key=lambda x: len(x[1]))[0]
            if not chosen:
                print(group_tracker)
                raise ValueError("Empty string")
            return chosen
        return "NA"


def label_terms() -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    with open(here("snakemake", "seq_analysis", "env.yaml")) as f:
        env = process_yaml(f)
        metadata: pl.DataFrame = read_tabular(METADATA_FILE)
    id_col = env["metadata"]["sample_col"]
    go_col = metadata["Gene Ontology IDs"].str.split(";")
    ggs = get_groups()
    tmp = {id_col: metadata[id_col]}
    for ns, groups in ggs.items():
        obj = GoGroup(G=GS, ns=ns, groups=groups)
        mapped = go_col.map_elements(
            lambda x: obj.label(map(lambda s: s.strip(), set(x))),
            return_dtype=pl.String,
        )
        tmp[ns] = mapped
    label_df = pl.DataFrame(tmp)
    count_dfs = {
        ns: label_df[ns].value_counts()
        for ns in ["BP", "CC", "MF"]
        if ns in label_df.columns
    }
    return label_df, count_dfs


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--update", action="store_true")
    parser.add_argument("-l", "--label", action="store_true")
    parser.add_argument("-g", "--group_info", action="store_true")
    args = vars(parser.parse_args())  # convert to dict
    return args


# BUG: there is some label that is the empty string
# TODO: alternative labelling approach is unsupervised clustering
# of sequences by annotation overlap and manually label

if __name__ == "__main__":
    args = parse_args()
    if args["update"]:
        update_candidate_files()
    elif args["group_info"]:
        groups = get_groups()
        for k, v in groups.items():
            print(f"{k}: {len(v)}")
    elif args["label"]:
        labelled, counts = label_terms()
        labelled.write_csv(WD / "labelled.csv")
        for ns, DF in counts.items():
            DF.write_csv(WD / f"{ns}_label_counts.csv")
