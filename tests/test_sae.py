from collections.abc import Sequence

import numpy as np
import polars as pl
import torch
from amr_predict.evaluation import EvalSAE, SaeMetrics
from attrs import define
from beartype import beartype
from torch import Tensor

RNG = np.random.default_rng()

SEP: str = ";"


N: int = 1000
CHOICES = list("abcdef")


@beartype
def gen_labels(
    n: int, together: dict[int, tuple[list, float]] | None = None
) -> pl.Series:
    """
    Randomly generate label sets

    Parameters
    ----------
    together : dict[int, list] | None
        Dictionary mapping indices of CHOICES to a tuple of
        other indices which must co-occur go together, followed by the
        co-occurence proportion
        e.g. {1: ([0, 2], 0.8)} means that CHOICES[1] co-occurs with
        CHOICES[0] and CHOICES[2] 80% of the time
    """

    def get():
        if not together:
            n_lab = RNG.integers(size=1, low=1, high=len(CHOICES), endpoint=True)
            return SEP.join(RNG.choice(CHOICES, size=n_lab))
        chosen = RNG.integers(size=1, low=0, high=len(CHOICES)).item()
        if chosen not in together:
            return CHOICES[chosen]
        idx, prop = together[chosen]
        others = [CHOICES[i] for i in idx]
        if RNG.random(1) <= prop:
            return SEP.join([CHOICES[chosen]] + others)
        return CHOICES[chosen]

    return pl.Series([get() for _ in range(n)])


@define
class DummyLatent:
    threshold: float
    labs: list[str]
    fire_prop: float = 1.0

    def activation(self, labels: pl.Series) -> Tensor:
        n = len(labels)
        dead = torch.distributions.Uniform(0, self.threshold).sample((n,))
        active = torch.distributions.Uniform(self.threshold, 1).sample((n,))
        has_label = labels.str.split(SEP).list.set_intersection(
            self.labs
        ).list.len() == len(self.labs)
        result = torch.where(torch.tensor(has_label), active, dead)
        if self.fire_prop < 1:
            mask_out = RNG.choice(range(n), size=int(self.fire_prop * n))
            result[mask_out] = dead[mask_out]
        return result


def test_multi_label(
    n: int = 1000,
    fire_prop: float = 0.9,
    threshold=0.5,
):
    """
    For each label, a latent will be generated that fires specifically
        on that label. Naturally it will also fire on samples where that
    label is also present

    Goal will be to see if the metrics can distinguish between each latent
    """
    pass


# * Case 1: Monosemantic latents, one for each label


def c1():
    labels = gen_labels(N, together={0: ([1, 2], 0.8), 3: ([4], 0.9)})
    latents: dict = {c: DummyLatent(threshold=0.5, labs=[c]) for c in CHOICES}
    activations = torch.hstack(
        [l.activation(labels).reshape(-1, 1) for l in latents.values()]
    )
    print(activations)
    eva = EvalSAE(activations, threshold=0.5)
    df = pl.DataFrame({"labels": labels, "sample": range(N)})
    metrics = eva.score_latents(labels=df, label_col="labels", label_sep=SEP)
    return metrics


# [2026-06-17 Wed] BUG: the values are weird, need to keep testing this
# sensitivity and specificity should be 100% by how the data are generated,
vals = c1()
report = vals.report(k=2, by="sensitivity")
