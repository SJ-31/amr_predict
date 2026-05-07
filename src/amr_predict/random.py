#!/usr/bin/env ipython

from __future__ import annotations

from typing import ClassVar

import amr_predict.aa as aa
import amr_predict.enums as ae
import dnachisel as dc
import numpy as np
import polars as pl
from attrs import Factory, define, field, validators
from numpy.random import Generator
from skbio.sequence import SubstitutionMatrix


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
                    .with_columns(pl.col("count").round_sig_figs(3))
                    # For numerical stability
                    .filter(pl.col(seq_col).is_in(self.token_choices))
                )
                counts = counts.with_columns(pl.col("count") / pl.col("count").sum())
                indices = np.array(
                    [counts[seq_col].index_of(c) for c in self.token_choices]
                )
                p = counts["count"][indices]
                assert p.sum() == 1, f"Counts should sum to 1..., got {p.sum()} instead"
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


@define
class BySubstitutionMatrix(Perturber):
    method = ae.Perturbations.SUBSTITUTION_MATRIX

    by_properties: bool = False
    # dynamic_matrix: bool = False
    # TODO: If true, determine which matrix to use for scoring based on...
    #
    k: int = 5
    # Choose the top k characters as potential replacements
    rng: Generator = field(default=None, converter=lambda x: np.random.default_rng(x))
    replacement_rate: float = field(
        default=0.5, validator=validators.and_(validators.le(1), validators.gt(0))
    )
    matrix: str = field(
        default="BLOSUM90",
        validator=validators.or_(
            validators.in_(SubstitutionMatrix.get_names()),
            validators.in_(aa.PHYLO_MATRICES.keys()),
        ),
    )
    mat: SubstitutionMatrix = field(
        init=False,
        default=Factory(
            lambda self: SubstitutionMatrix.by_name(self.matrix)
            if self.matrix in SubstitutionMatrix.get_names()
            else aa.read_paml(aa.PHYLO_MATRICES[self.matrix]),
            takes_self=True,
        ),
    )
    choices: dict = field(init=False, factory=dict)

    def substitute(self, char: str) -> str:
        if char not in self.choices and not self.by_properties:
            score_sort = sorted(self.mat.to_dict()[char].items(), key=lambda x: x[1])
            self.choices[char] = list(map(lambda x: x[0], score_sort))[: self.k]
        elif char not in self.choices:
            clst = aa.AA_DF.filter(pl.col("aa") == char)["cluster"]
            self.choices[char] = aa.AA_DF.filter(pl.col("cluster") == clst)[
                "cluster"
            ].to_list()
        choices: list[str] = self.choices[char]
        return self.rng.choice(choices, size=1).item()

    def __call__(self, sequence: str) -> str:
        indices = range(len(sequence))
        to_replace: np.ndarray = self.rng.choice(
            indices, size=int(len(indices) * self.replacement_rate)
        )
        chars = list(sequence)
        for idx in to_replace:
            current = chars[idx]
            chars[idx] = self.substitute(current)
        return "".join(chars)
