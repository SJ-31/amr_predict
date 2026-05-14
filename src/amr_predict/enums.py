#!/usr/bin/env python3

from enum import Enum

from attrs import define


class RandomizationMethods(Enum):
    PERMUTATION = "permutation"
    DENOVO = "denovo"


class SeqCovariates(Enum):
    functional_similarity = "functional_similarity"
    sequence_similarity = "sequence_similarity"
    taxonomic_similarity = "taxonomic_similarity"


@define
class PerturbationCfg:
    species: str = "h_sapiens"
    optimization_method: str = "use_best_codon"


class Perturbations(Enum):
    CODON_OPTIMIZATION = "codon_optimization"
    SUBSTITUTION_MATRIX = "substitution_matrix"


SeqTypes = Enum("SeqTypes", (("AA", "aa"), ("NUC", "nuc")))

BasicPoolings = Enum(
    "BasicPoolings",
    {n.upper(): n for n in ["max", "sum", "mean", "similarity", "cls"]},
)

# * Embeddings
EsmSynthraModels = Enum(
    "EsmSynthraModels",
    {
        "esmc_600m_synthyra": "Synthyra/ESMplusplus_large",
        "esmc_300m_synthyra": "Synthyra/ESMplusplus_small",
    },
)


class OmniNaModels(Enum):
    omniNA_66m = "XLS/OmniNA-66m"
    omniNA_220m = "XLS/OmniNA-220m"
    omniNA_1p7b = "XLS/OmniNA-1.7b"


EsmModels = Enum(
    "EsmModels",
    {
        i: i
        for i in [
            "esm2_t6_8m_UR50D",
            "esm2_t33_650m_UR50D",
            "esm3_open",
            "esmc_600m",
            "esmc_300m",
        ]
    },
)


SeqLensModels = Enum(
    "SeqLensModels", {"seqLens_4096_512_46M_Mp": "omicseye/seqLens_4096_512_46M-Mp"}
)

EmbeddingModels = Enum(
    "EmbeddingModels",
    dict(
        {i.name: i.value for i in EsmSynthraModels}
        | {i.name: i.value for i in SeqLensModels}
        | {i.name: i.value for i in EsmModels}
        | {i.name: i.value for i in OmniNaModels}
    ),
)
