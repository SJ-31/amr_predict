#!/usr/bin/env ipython

from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl
import polars.selectors as cs
from attrs import Factory, define, field

# Taxonomic distance measured using shortest path between nodes
# (basically the patristic distance but without summing)
# https://scikit.bio/docs/latest/generated/skbio.tree.TreeNode.distance.html#skbio.tree.TreeNode.distance
# Would like to use patristic distance or other measures that take into
# account branch length, but due to the scale of the ncbi taxonomy database,
# it is unfeasable to construct such a structure
# Instead you will create an unweighted (branch lengths equal) taxonomy
# tree to use to get shortest paths. Distances between taxa will also
# be weighted by the taxonomic rank they share e.g. same genus gets
# higher similarity than same family


NCBI_TAX_SCHEMA: dict[str, pl.DataType] = {
    "tax_id": pl.UInt32,
    "parent_tax_id": pl.UInt32,
    "rank": pl.String,
    "embl_code": pl.String,
    "division_id": pl.UInt16,
    "inherited_div_flag": pl.Boolean,
    "genetic_code_id": pl.UInt8,
    "inherited_gc_flag": pl.Boolean,
    "mitochondrial_genetic_code_id": pl.UInt8,
    "inherited_mgc_flag": pl.Boolean,
    "genbank_hidden_flag": pl.Boolean,
    "hidden_subtree_root_flag": pl.Boolean,
    "comments": pl.String,
}


def get_ranks_mapping():
    ranks = [
        "acellular root",
        "cellular root",
        "domain",
        "kingdom",
        "subkingdom",
        "superphylum",
        "phylum",
        "subphylum",
        "superclass",
        "class",
        "subclass",
        "infraclass",
        "cohort",
        "subcohort",
        "superorder",
        "order",
        "suborder",
        "infraorder",
        "parvorder",
        "superfamily",
        "family",
        "subfamily",
        "tribe",
        "subtribe",
        "genus",
        "subgenus",
        "section",
        "subsection",
        "series",
        "species group",
        "species subgroup",
        "species",
        "subspecies",
        "varietas",
        "subvariety",
        "forma",
        "forma specialis",
        "strain",
        "isolate",
        "biotype",
        "genotype",
        "morph",
        "pathogroup",
        "serogroup",
        "serotype",
    ]
    return {r: len(ranks) - i for i, r in enumerate(ranks)}


RANK2LEVEL: dict[str, int] = get_ranks_mapping()


@define
class TaxonomyTree:
    file: str | None = None
    G: nx.DiGraph = field(
        init=False,
        default=Factory(
            lambda self: self.digraph_from_file(self.file), takes_self=True
        ),
    )
    # NOTE: DiGraph's measurement of shortest paths is direction-aware
    # and won't go up parent nodes
    rank_weights: dict = field(factory=lambda: {"species": 9})

    def species_or_lower(self, id: int) -> bool:
        rank: str = self.G.nodes[id]["rank"]
        return rank in {
            "species",
            "subspecies",
            "varietas",
            "subvariety",
            "forma",
            "forma specialis",
            "strain",
            "isolate",
            "biotype",
            "genotype",
            "morph",
            "pathogroup",
            "serogroup",
            "serotype",
        }

    def dist(self, a: int, b: int) -> int:
        lca: int = fast_lca(self.G, a, b)
        if self.species_or_lower(lca):
            return 0
        lca_rank: str = self.G.nodes[lca]["rank"]
        return RANK2LEVEL[lca_rank] - RANK2LEVEL["species"]

    @staticmethod
    def digraph_from_file(file: str | Path) -> nx.DiGraph:
        """
        Construct digraph from NCBI `nodes.dmp` file or GML version of
        it
        """
        file = Path(file) if isinstance(file, str) else file
        if file.suffix == ".dmp":
            edges_to_add = []
            G = nx.DiGraph()
            lf = (
                pl.scan_csv(
                    file,
                    separator="|",
                    new_columns=list(NCBI_TAX_SCHEMA.keys()) + ["_ext"],
                )
                .with_columns(cs.string().str.strip_chars())
                .drop("_ext")
                .cast({"tax_id": pl.UInt32, "parent_tax_id": pl.UInt32})
            )
            for row in lf.collect().iter_rows(named=True):
                G.add_node(row["tax_id"], rank=row["rank"])
                edges_to_add.append((row["parent_tax_id"], row["tax_id"]))
            G.add_edges_from(edges_to_add)
        else:
            G = nx.read_gml(file)
            if not all(("rank" in node[1] for node in G.nodes.items())):
                raise ValueError("Nodes must have the `rank` attribute")
        if not nx.is_directed(G):
            raise nx.NetworkXError("G must be a DAG")
        elif len(G) == 0:
            raise nx.NetworkXPointlessConcept("G is empty")
        return G


def fast_lca(D: nx.DiGraph, a: int, b: int) -> int:
    # TODO: this method assumes every taxa share a common ancestor
    # that is under a canonical taxonomic rank. Is this correct?
    a_ancestors, b_ancestors = nx.ancestors(D, a), nx.ancestors(D, b)
    common = set(a_ancestors) & set(b_ancestors)
    if not common:
        return -1
    else:
        with_levels = [
            (RANK2LEVEL.get(D.nodes[r].get("rank", ""), np.inf), r) for r in common
        ]
    return min(with_levels, key=lambda x: x[0])[1]
