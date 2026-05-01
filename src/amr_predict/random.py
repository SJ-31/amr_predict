#!/usr/bin/env ipython

from __future__ import annotations

from typing import ClassVar

import amr_predict.enums as ae
import dnachisel as dc
import numpy as np
import polars as pl
from attrs import Factory, define, field
from numpy.random import Generator


@define
class Randomizer:
    """
    Helper class for randomizing sequences for evaluation tasks.

    If with_distribution, the observed token distribution in the passed dataset is used
        to randomly generate sequences
    """

    method: ae.RandomizationMethods
    rng: Generator = field(default=None, converter=lambda x: np.random.default_rng(x))
    seqtype: ae.SeqTypes = ae.SeqTypes.NUC
    with_distribution: bool = False
    token_choices: tuple = field(
        init=False,
        default=Factory(
            lambda self: list("ATCG")
            if self.seqtype == ae.SeqTypes.NUC
            else list("ARNDCEQGHILKMFPSTWYV"),
            takes_self=True,
        ),
    )

    def _generate_random(self, seq: str, p: np.ndarray | None = None) -> str:
        if p is None:
            generated = np.random.choice(
                self.token_choices, size=len(seq), replace=True
            )
        else:
            generated = np.random.choice(
                self.token_choices, size=len(seq), replace=True, p=p
            )
        return "".join(generated)

    def _permute_sequence(self, seq: str) -> str:
        return "".join(self.rng.permutation(list(seq)))

    def randomize(
        self, dataset: pl.DataFrame, seq_col: str = "sequence"
    ) -> pl.DataFrame:
        if self.method == ae.RandomizationMethods.PERMUTATION:
            return dataset.with_columns(
                pl.col(seq_col).map_elements(
                    self._permute_sequence, return_dtype=pl.String
                )
            )
        elif self.method == ae.RandomizationMethods.DENOVO:
            if self.with_distribution:
                counts = (
                    dataset.with_columns(pl.col(seq_col).str.split(""))
                    .explode(seq_col)[seq_col]
                    .value_counts()
                )
                counts = counts.with_columns(pl.col("count") / pl.col("count").sum())
                assert counts["count"].sum() == 1, "Counts should sum to 1..."
                dist = {c: f for c, f in counts.iter_rows()}
                p = np.array([dist[char] for char in self.token_choices])
            else:
                p = None
            return dataset.with_columns(
                pl.col(seq_col).map_elements(
                    lambda x: self._generate_random(x, p), return_dtype=pl.String
                )
            )
        raise NotImplementedError()


@define
class Perturber:
    method: ClassVar[ae.Perturbations | None] = None
    _registry: ClassVar[dict[ae.Perturbations, Perturber]] = {}
    seqtype: ae.SeqTypes = ae.SeqTypes.NUC
    cfg: ae.PerturbationCfg = field(factory=ae.PerturbationCfg)

    def __call__(self, sequence: str) -> str:
        raise NotImplementedError()

    @classmethod
    def __attrs_init_subclass__(cls):
        if cls.method is not None:
            Perturber._registry[cls.method.name] = cls
            Perturber._registry[cls.method] = cls

    def perturb(self, dataset: pl.DataFrame, seq_col: str = "sequence") -> pl.DataFrame:
        return dataset.with_columns(
            pl.col(seq_col).map_elements(self.__call__, return_dtype=pl.String)
        )

    @classmethod
    def new(_cls, method: ae.Perturbations, **kws):
        return Perturber._registry[method](**kws)


@define
class CodonOptimizer(Perturber):
    method = ae.Perturbations.CODON_OPTIMIZATION

    def __call__(self, sequence: str) -> str:
        problem = dc.DnaOptimizationProblem(
            sequence=sequence,
            constraints=[dc.EnforceTranslation()],
            objectives=[
                dc.CodonOptimize(
                    species=self.cfg.species,
                    method=self.cfg.optimization_method,
                )
            ],
        )
        problem.resolve_constraints()
        problem.optimize()
        return problem.sequence


# @define
# class
# TODO: then amino acid physicochemical properties
# https://pmc.ncbi.nlm.nih.gov/articles/PMC11884779/#evaf025-s2
# https://www.science.org/doi/10.1126/sciadv.aax3124
