#!/usr/bin/env python3
import pickle
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import polars as pl
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from peptides import Peptide

if TYPE_CHECKING:
    from snakemake.iocontainers import snakemake
PARAMS: dict = snakemake.params
RCONFIG: dict = snakemake.config.get(snakemake.rule, {})
RNG: int = snakemake.config.get("rng", 20021031)
INPUT = snakemake.input
OUTPUT = snakemake.output
WC = snakemake.wildcards


def translate(seq: SeqRecord | Seq) -> Peptide:
    if " " in seq:
        seq = seq.replace(" ", "")
    if (remainder := len(seq) % 3) != 0:
        add = 3 - remainder
        seq = seq + ("N" * add)
    translated = seq.translate(id=True, name=True, description=True)
    return Peptide(str(translated.seq))


def parse_k2_lca(lca: str, classified_taxid: int, G: nx.DiGraph):
    """
    Parse kraken2's LCA string to count the number of k-mers assigned
    to the classified taxon (and its descendants), k-mers not found
    in the database, k-mers assigned to other taxa, and k-mers with
    ambiguous nucleotides

    Parameters
    ----------
    lca : str
    classified_taxid : int
    G : nx.DiGraph
        DiGraph storing ncbi taxonomy relations
    """
    descendants = [str(d) for d in nx.descendants(G, classified_taxid)]
    result: dict = {
        "kmer_ambiguous": 0,
        "kmer_unknown": 0,
        "kmer_other_taxid": 0,
        "kmer_total": 0,
        "kmer_direct": 0,
        "n_other_taxid": 0,
        "kmer_other_taxid_std": None,
    }
    mapping = {"A": "kmer_ambiguous", "0": "kmer_unknown"}
    descendants.append(str(classified_taxid))
    mapping.update({d: "kmer_direct" for d in descendants})
    counts: pl.DataFrame = (
        pl.DataFrame(
            map(lambda x: x.split(":"), lca.split(" ")),
            schema={"taxid": str, "count": int},
        )
        .with_columns(pl.col("taxid"))
        .group_by("taxid")
        .agg(pl.col("count").sum())
    )
    agg: dict = (
        counts.with_columns(
            pl.col("taxid")
            .replace_strict(mapping, default="kmer_other_taxid")
            .alias("category")
        )
        .group_by("category")
        .agg(pl.col("count").sum(), pl.col("count").std().alias("std"))
        .rows_by_key("category", unique=True, named=True)
    )
    result.update({k: v["count"] for k, v in agg.items()})
    result["kmer_total"] = counts["count"].sum()
    result["kmer_other_taxid_std"] = agg["kmer_other_taxid"].get("std")
    result["n_other_taxid"] = len(
        set(counts["taxid"].unique()) - set(descendants) - {"0"}
    )
    return result


def get_nearest_dist(file: Path | str, prefix: str = "query_") -> pl.DataFrame:
    """
    For each generated sequence, return the distance to the nearest
    entries in `dist_df` and their identities
    """
    import polars.selectors as cs

    dist_df = pl.read_csv(file, separator="\t", skip_rows=1, has_header=False)
    dist_df.columns = ["node"] + dist_df["column_1"].to_list()

    no_queries = dist_df.filter(
        ~pl.col("node").str.starts_with(prefix)
    ).with_row_index()
    min_idx: pl.DataFrame = no_queries.select(
        cs.starts_with("query").arg_min()
    ).transpose(include_header=True, header_name="query", column_names=["index"])
    min_vals: pl.DataFrame = no_queries.select(cs.starts_with("query").min()).transpose(
        include_header=True, header_name="query", column_names=["value"]
    )
    return (
        min_idx.join(no_queries.select(["node", "index"]), on="index")
        .join(min_vals, on="query")
        .drop("index")
    )


def get_props(
    peps: list[Peptide], descriptor: str, ids: list[str] | None = None, **kws
) -> pl.DataFrame:
    if descriptor not in dir(peps[0]):
        raise ValueError(f"the given descriptor `{descriptor}` isn't valid")

    r1 = getattr(peps[0], descriptor)(**kws)
    ids = [f"p{i}" for i in range(len(peps))] if ids is None else ids
    convert = "_fields" in dir(r1)

    def fn(pep: Peptide) -> np.ndarray | list:
        res = getattr(pep, descriptor)(**kws)
        if convert:
            return np.array(res)
        return res

    tmp = [r1] + [fn(pep) for pep in peps[1:]]
    df = (
        pl.DataFrame(np.vstack(tmp), schema=r1._fields)
        if convert
        else pl.DataFrame({descriptor: tmp})
    )
    df = df.with_columns(pl.Series(ids).alias("id"))
    return df


def repetition_score(seq: str) -> dict:
    """ """
    pass


# * Rules


def translate_fasta() -> None:
    with open(OUTPUT[0], "w") as f:
        gen = [
            f">{seq.id}\n{translate(seq).sequence}"
            for seq in SeqIO.parse(INPUT[0], "fasta")
        ]
        f.write("\n".join(gen))


def get_repetition_score() -> None:
    pass


def describe_protein_seqs() -> None:
    """
    Compute the generated sequences' average distance from their prompt
    across several peptide descriptors
    """
    from functools import reduce

    from scipy.spatial.distance import cdist

    prompt: SeqRecord = SeqIO.read(INPUT["prompt_full"], "fasta")
    ids, peps = [prompt.id], [Peptide(str(prompt.seq))]

    for seq in SeqIO.parse(INPUT["generated"], "fasta"):
        ids.append(seq.id)
        peps.append(Peptide(str(seq.seq)))

    tmp_dist = {"descriptor": [], "value": []}
    dfs = []
    for descriptor, kws in RCONFIG.items():
        kws = kws or {}
        df = get_props(peps=peps, descriptor=descriptor, ids=ids, **kws)
        dfs.append(df)

        prompt_val: np.ndarray = np.array([df.drop("id").row(0)])
        vals: np.ndarray = df.drop("id").slice(1).to_numpy()
        tmp_dist["descriptor"].append(descriptor)
        tmp_dist["value"].append(cdist(prompt_val, vals).mean())

    mean_dist: pl.DataFrame = pl.DataFrame(tmp_dist)
    combined: pl.DataFrame = reduce(lambda x, y: x.join(y, on="id"), dfs)
    mean_dist.write_csv(OUTPUT["mean"])
    combined.write_csv(OUTPUT["vals"])


def sort_k2_hits():
    dfs = []
    with open(PARAMS["digraph"], "rb") as f:
        G: nx.DiGraph = pickle.loads(f)
    for i in INPUT:
        db = Path(input).parent.stem
        df = pl.read_csv(
            i,
            separator="\t",
            schema={"status": str, "id": str, "taxid": int, "length": int, "lca": str},
        ).with_columns(pl.lit(db).alias("query_db"))
        df = df.with_columns(
            pl.Series(
                parse_k2_lca(lca, assigned, G)
                for lca, assigned in zip(df["lca"], df["taxid"])
            ).struct.unnest()
        )
        dfs.append(df)
    df = pl.concat(dfs, how="diagonal_relaxed")
    df.write_csv(OUTPUT[0])


def sort_mmseqs_hits():
    """
    Sort by evalue then bitscore ranking i.e. pick sequences with the lowest evalue,
    (likely to have multiple sequences with the same evalue)
    then get the best hit by bitscore
    """
    dfs = []
    for top in INPUT:
        db = Path(input).parent.stem
        df = pl.read_csv(top, separator="\t").with_columns(
            rank=pl.col("evalue").rank().over("query")
        )
        best_evalue = df.group_by("query").agg(min_rank=pl.col("rank").min())
        df = (
            df.join(best_evalue, on="query")
            .filter(pl.col("rank") == pl.col("min_rank"))
            .drop(["rank", "min_rank"])
            .group_by("query")
            .agg(pl.all().top_k_by(k=1, by="bits"))
            .explode(pl.all().exclude("query"))
            .with_columns(pl.lit(db).alias("query_db"))
        )
        dfs.append(df)
    pl.concat(dfs, how="diagonal relaxed").write_csv(OUTPUT[0])


def fmt_prompts():
    header: str = WC["prompt"]
    data: dict = PARAMS["prompt2data"][header]
    prompt_full = data["file_full"]

    if data["file"]:
        prompt_seq = str(SeqIO.read(data["file"], "fasta").seq)
    else:
        prompt_seq = data["seq"]
    with open(OUTPUT["prompt"], "w") as f:
        f.write(f">{header}\n{prompt_seq}")

    full_seq = SeqIO.read(prompt_full, "fasta")
    trimmed = SeqRecord(Seq(str(full_seq.seq).removeprefix(prompt_seq)))
    if data.get("prompt_from_full", True):
        assert (
            str(trimmed.seq) != str(full_seq.seq)
        ), "ERROR: could not trim full sequence. Ensure prompt is 100% derived from full sequence"

    for seq, suffix in zip((full_seq, trimmed), ("", "_trimmed")):
        translated = translate(seq).sequence
        if suffix:
            sf = suffix.replace("_", "-").upper()
        else:
            sf = "-FULL"
        Path(OUTPUT[f"full{suffix}"]).write_text(f">{header}{sf}\n{str(seq.seq)}")
        Path(OUTPUT[f"aa{suffix}"]).write_text(f">{header}{sf}\n{translated}")


def extract_distances():
    from Bio import Phylo

    min_dist: pl.DataFrame = get_nearest_dist(
        INPUT["dist"], prefix=PARAMS["query_prefix"]
    ).rename({"node": "nearest", "value": "distance_to_nearest"})
    tree = Phylo.read(INPUT["tree"], "newick")
    assert tree.rooted, "Tree must be rooted"
    min_dist = min_dist.with_columns(
        pl.col("query")
        .map_elements(lambda x: tree.distance(x), return_dtype=pl.Float64)
        .alias("distance_to_root")
    )
    min_dist.write_csv(OUTPUT[0])


if rule_fn := globals().get(snakemake.rule):
    rule_fn()
