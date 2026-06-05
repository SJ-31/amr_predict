#!/usr/bin/env ipython

from operator import index

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
from amr_predict.enums import SeqCovariates
from amr_predict.metrics import (
    EmbeddingCorrelations,
    NeighborMetrics,
    generalized_mc_test,
    mc_multinomial_test,
)
from loguru import logger
from numpy.random import Generator

logger.enable("amr_predict")


@pytest.fixture
def random_dset(rng: Generator) -> ad.AnnData:
    n = 500
    x = rng.random((n, 20))
    cats = {"a": list("ABCDEFGH"), "b": range(10)}
    annots = [f"a{i}" for i in range(10)]
    meta = pd.DataFrame(
        {
            "a": [rng.choice(cats["a"], 1).item() for _ in range(n)],
            "b": [rng.choice(cats["b"], 1).item() for _ in range(n)],
            "anno": [
                ";".join(rng.choice(annots, rng.integers(1, 5, size=1)))
                for _ in range(n)
            ],
        }
    )
    return ad.AnnData(X=x, obs=meta)


def multinomial_setup(
    rng: Generator,
    n_categories: int = 9,
    dist_size=10_000,
    n: int = 5,
    obs_limit: int | None = None,
):
    from mimesis import Text

    txt = Text()
    choices = txt._dataset[:n_categories]

    dist = [rng.choice(choices) for _ in range(dist_size)]

    def fn():
        if not obs_limit:
            obs = rng.choice(choices, size=n, replace=True)
        else:
            obs = rng.choice(choices[:obs_limit], size=n, replace=True)
        return mc_multinomial_test(obs, n=n, all_categories=dist, seed=rng)

    return fn


@pytest.mark.parametrize(
    "pairable,ssprop,level",
    [
        (False, None, "seqs"),
        (True, None, "seqs"),
        (False, {"natural": 0.5, "random": 0.8, "perturbed": 0.5}, "seqs"),
        (False, {"natural": 0.5, "random": 0.8, "perturbed": 0.5}, "tokens"),
        (False, None, "tokens"),
    ],
)
def test_perturbation_metrics(pairable, ssprop, level, random_linked_dset):
    from amr_predict.metrics import PerturbationMetrics

    n = 1000
    natural = random_linked_dset(n)
    random = random_linked_dset(n)
    pert = random_linked_dset(n)

    P = PerturbationMetrics(
        natural=natural,
        random=random,
        perturbed=pert,
        random_is_pairable=pairable,
        level=level,
        id_col="id",
        embedding_distance="euclidean",
        subsample_prop=ssprop,
        seed=942,
    )
    assert P.idx_df.height == 1000
    result = P.run()
    print(result)
    print(P.idx_df)


# TODO: use this to check how sensitive it is


def test_multinomial(rng):
    from mimesis import Text

    txt = Text()
    dist = [txt.color() for _ in range(10000)]

    obs = txt._dataset["color"]

    logger.info("dist {}", pl.Series(dist).value_counts())

    f1 = mc_multinomial_test(obs, n=len(obs), all_categories=dist, seed=rng)
    assert f1.p_adj >= 0.05
    logger.info("random {}", f1.p_adj)

    f1 = mc_multinomial_test(
        ["Beige"] * len(obs), n=len(obs), all_categories=dist, seed=rng
    )
    logger.info("real {}", f1.p_adj)
    assert f1.p_adj <= 0.05

    biased = rng.choice(
        obs, size=len(obs), replace=True, p=[0.25, 0.25, 0.25, 0.25] + [0] * 9
    )
    b = mc_multinomial_test(biased, n=len(obs), all_categories=dist, seed=rng)
    logger.info("biased {}, {}", biased, b.p_adj)
    assert b.p_adj <= 0.05


def test_category_mat(rng):
    n = 100
    choices = np.array(range(n))
    cats = rng.choice(["a", "b", "c", "d", "e"], size=n, replace=True)
    indices = rng.choice(choices, size=5)
    n_neighors = 8

    neighbors = np.array(
        [rng.choice(choices, size=n_neighors, replace=False) for _ in range(n)]
    )
    assert neighbors.shape[1] == n_neighors

    enc, mat, coded_cats = NeighborMetrics._get_category_matrix(
        cats, neighbors, indices
    )
    logger.info("cats {}", cats)
    logger.info("neighbors {}", neighbors)
    logger.info("mat {}, coded {}", mat, coded_cats)
    assert (coded_cats[indices] == mat[:, 0].numpy()).all()
    for i in range(n_neighors):
        assert (coded_cats[neighbors[indices, i]] == mat[:, i + 1].numpy()).all()


def test_neighbor_metrics_rand(random_linked_dset):
    met = NeighborMetrics(
        dset=random_linked_dset(n=5000),
        category_cols=["c1", "c2"],
        anno_cols=["a1"],
        n_resample=10_000,
        n_resample_chi_square=1000,
        n_neighbors=10,
    )
    df, dist = met.run(with_randomization=True)
    df.write_csv("./neighbor_metrics_rand.csv")
    logger.info("{}", df)
    assert (df["p_value"] > 0.05).all()


# TODO:  Parameters to check for neighbors: number of neighbors, number of categories


def test_corr_rand(random_linked_dset):
    met = EmbeddingCorrelations(
        dset=random_linked_dset(n=1000),
        embedding_distance="cosine",
        columns={
            "a1": SeqCovariates.functional_similarity,
            "s1": SeqCovariates.sequence_similarity,
        },
        n_resample=1000,
    )
    df = met.run()
    print(df)


@pytest.mark.parametrize(
    "loc,sig,alt", [(0, False, "two-sided"), (4, True, "greater"), (-4, True, "less")]
)
def test_mc(loc, sig, alt, rng: Generator):
    data = torch.tensor(rng.normal(loc=loc, size=100))
    result = generalized_mc_test(
        data,
        statistic=lambda x: x,
        rvs=lambda: torch.tensor(rng.normal(size=1000)),
        vectorized=True,
        alternative=alt,
    )
    assert (result.p_values >= 0).all()
    assert (result.p_values <= 1).all()
    if not sig:
        assert 1 >= result.p_adj >= 0.05
    else:
        assert 0 <= result.p_adj <= 0.05


def harmonic_mean_pvalue(
    p_values: np.ndarray, weights: np.ndarray | None = None, alpha: float | None = None
):
    """
    Test the null hypothesis that none of the p values in `p_values` is significant,
    without assuming any dependency structure between them [1]

    Notes
    -----
    This implementation assumes is for stand-alone tests

    'for small values (e.g. below 0.05),
    the HMP can be directly interpreted as a p-value after
    adjusting for multiple comparison'

    [1]
    """
    pass
    # TODO: download the package from
    # https://cran.r-project.org/web/packages/harmonicmeanp/index.html
    # and use the p.hmp function (see )
    # https://cran.r-project.org/web/packages/harmonicmeanp/vignettes/harmonicmeanp.html
    # but check if normal R packages work in container env
    # Backup is to use bonferroni
