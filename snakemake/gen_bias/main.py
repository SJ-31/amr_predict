#!/usr/bin/env python3
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

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
    if (remainder := len(seq) % 3) != 0:
        add = 3 - remainder
        seq = seq + ("N" * add)
    translated = seq.translate(id=True, name=True, description=True)
    return Peptide(str(translated.seq))


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
        print(gen)
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
