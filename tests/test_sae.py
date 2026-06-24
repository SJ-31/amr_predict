from collections.abc import Sequence
from functools import reduce

import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from amr_predict.evaluation import EvalSAE, SaeMetrics
from attrs import define
from beartype import beartype
from sklearn.metrics import accuracy_score
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


# * Case 1: Monosemantic latents, one


def tester(
    label_args: dict,
    true_firing: dict[str, list[str]],
    threshold: float = 0.3,
    fire_prop=1.0,
):
    latents = {
        key: DummyLatent(threshold=threshold, labs=v, fire_prop=fire_prop)
        for key, v in true_firing.items()
    }
    k = max(map(len, true_firing.values()))
    labels = gen_labels(N, **label_args)
    activations = torch.hstack(
        [l.activation(labels).reshape(-1, 1) for l in latents.values()]
    )
    eva = EvalSAE(activations, threshold=threshold, lidx=true_firing.keys())
    df = pl.DataFrame({"labels": labels, "sample": range(N)})
    metric_obj = eva.score_latents(
        labels=df,
        label_col="labels",
        label_sep=SEP,
        normalize=False,
    )
    metrics = [
        "accuracy",
        "mcc",
        "negative_predictive_value",
        "precision",
        "sensitivity",
        "specificity",
    ]
    template = {"metric": [], "latent_idx": [], "acc": []}
    reports = []
    for m in metrics:
        report = metric_obj.report(k=k, by=m)
        to_append = report.with_columns(pl.lit(m).alias("by"), cs.list().list.join(","))
        reports.append(to_append)
        for latent, truth in true_firing.items():
            top_labels = report.filter(pl.col("latent_idx") == latent)["label"][0]
            if isinstance(top_labels, str):
                top_labels = [top_labels]
            else:
                top_labels = top_labels.to_list()[: len(truth)]
            acc = accuracy_score(y_true=truth, y_pred=top_labels)
            template["metric"].append(m)
            template["latent_idx"].append(latent)
            template["acc"].append(acc)
    metric_acc = pl.DataFrame(template)
    return pl.concat(reports, how="diagonal_relaxed"), metric_acc


# Only normalize if the real activations > 1 and less than 0

# [2026-06-17 Wed] TODO: this works, now you wanna see if you can
# recover the true latent label assignments
# under different cases
# TODO: Investigate which metric is most robust to co-occurence

label_params = {
    "mono": {
        "label_args": {"together": {0: ([1, 2], 0.8), 3: ([4], 0.9)}},
        "true_firing": {c: [c] for c in CHOICES},
    },
    "mono_perfect_co-occur": {
        "label_args": {"together": {0: ([1, 2], 1.0), 3: ([4], 1.0)}},
        "true_firing": {"l1": ["a"], "l2": ["d"], "l3": ["b"]},
    },
    "poly": {
        "label_args": {"together": {0: ([1, 2], 1.0), 3: ([4], 1.0)}},
        "true_firing": {"l1": CHOICES[:2], "l2": ["a"]},
    },
}

results = {}
for k, v in label_params.items():
    results[k] = tester(**v)
