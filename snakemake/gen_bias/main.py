#!/usr/bin/env python3
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from peptides import Peptide

if TYPE_CHECKING:
    from snakemake.iocontainers import snakemake
PARAMS: dict = snakemake.params
RCONFIG: dict = snakemake.config.get(snakemake.rule, {})
RNG: int = snakemake.config.get("rng", 20021031)


if rule_fn := globals().get(snakemake.rule):
    rule_fn()


def translate(seq: SeqRecord) -> Peptide:
    if (remainder := len(seq) % 3) != 0:
        add = 3 - remainder
        seq = seq + ("N" * add)
    translated = seq.translate(id=True, name=True, description=True)
    return Peptide(str(translated.seq))


def get_props(
    peps: list[Peptide], descriptor: str, ids: list[str] | None = None, **kws
) -> pl.DataFrame:
    if descriptor not in dir(peps[0]):
        raise ValueError(f"the given descriptor `{descriptor}` isn't valid")

    r1 = getattr(peps[0], descriptor)(**kws)
    ids = [f"p{i}" for i in range(len(peps))] if ids is None else ids
    convert = "_fields" in dir(r1)

    def fn(pep: Peptide) -> np.ndarray | list:
        res = getattr(pep, descriptor, **kws)()
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


def describe_protein_seqs() -> None:
    """
    Compute the generated sequences' average distance from their prefix
    across several peptide descriptors
    """
    from functools import reduce

    from peptides import Peptide

    prefix: SeqRecord = next(SeqIO.parse(PARAMS["prefix_fasta"], "fasta"))
    ids, peps = [prefix.id], [translate(prefix)]
    for seq in SeqIO.parse(input[0], "fasta"):
        ids.append(seq.id)
        peps.append(translate(seq))
    dfs = []
    for descriptor, kws in RCONFIG.items():
        kws = kws or {}
        df = get_props(peps=peps, descriptor=descriptor, ids=ids, **kws)
        dfs.append(df)
    combined: pl.DataFrame = reduce(lambda x, y: x.join(y, on="id"), dfs)
