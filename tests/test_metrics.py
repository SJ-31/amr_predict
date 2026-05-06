#!/usr/bin/env ipython

from operator import index

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from amr_predict.metrics import NeighborMetrics, generalized_mc_test
from numpy.random import Generator


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


def test_neighbor_metrics_rand(random_linked_dset):
    met = NeighborMetrics(
        dset=random_linked_dset(n=10_000),
        category_cols=["c1", "c2"],
        anno_cols=["a1"],
        n_resample=10000,
        n_neighbors=4,
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
