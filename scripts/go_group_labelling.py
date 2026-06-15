#!/usr/bin/env ipython

from collections.abc import Sequence
from typing import Literal

import networkx as nx
import polars as pl
import yaml
from attrs import define, field
from pyhere import here

wd = here("data", "meta", "GO")
wd.mkdir(exist_ok=True)
df = pl.read_csv(wd / "go_attrs.csv", null_values="NA")

G = nx.read_gml(wd / "go.gml", label="name")


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


def update_candidate_files():
    groups: dict = {}
    for ns in ["BP", "CC", "MF"]:
        group_file = wd / f"{ns}_groups.yaml"
        if group_file.exists():
            with open(group_file, "r") as f:
                groups[ns] = yaml.safe_load(f)
    in_group = [v for ns, g in groups.items() for k, vals in g.items() for v in vals]

    def has_paths_with_prev(x) -> bool:
        for node in in_group:
            # Source must be the potential child node in this format
            if nx.has_path(GS, source=x, target=node):
                return True
        return False

    dfs = {
        ns: df.filter(
            (~pl.any_horizontal([pl.col(b) > 0 for b in blacklist]))
            & (pl.col("maxDistanceToNs") <= 4)
            & (pl.col("namespace") == ns)
            # & (~pl.col("subset").str.contains("gocheck_do_not_annotate"))
            # & (pl.col("subset").is_not_null())
            & (~pl.col("name").is_in(in_group))
        )
        for ns in ["BP", "CC", "MF"]
    }
    print(dfs)
    dfs = {
        k: df.filter(
            ~pl.col("name").map_elements(
                lambda x: has_paths_with_prev(x), return_dtype=pl.Boolean
            )
        )
        for k, df in dfs.items()
    }
    for k, v in dfs.items():
        v.write_csv(wd / f"{k}_list.csv")


@define
class GoGroup:
    """
    Class for assigning GO groups to entities with sets of GO terms

    Parameters
    ----------
    cache : dict[tuple[str, str], bool]
        Dictionary caching whether a term (first element of key)
        is a child of terms in a GO group (second element of key)
    """

    G: nx.DiGraph
    ns: Literal["BP", "CC", "MF"]
    groups: dict[str, set]
    cache: dict[tuple[str, str], bool] = field(init=False, factory=dict)

    def assign(terms: Sequence[str]) -> str:
        """
        Assign a GO group (GG) to a set of GO terms

        If the set could be labelled as multiple GGs, choose the GG that
        collectively has the most children

        Break ties by...

        Filter out the sets of GO terms by their

        Returns
        -------
        A GO group name, a key of "groups"
        """
        pass


update_candidate_files()

# TODO: Group GO terms manually, but also in a later step ensure that
# terms in groups you name aren't children of each other.
# Call these GO groups (GG)
#
# If a sequence could be labelled as multiple GGs, choose the GG that
# collectively has the most children
# Filter out the sets of GO terms by their
