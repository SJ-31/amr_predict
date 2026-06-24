import copy

import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from amr_predict.evaluation import EvalSAE
from attrs import define
from beartype import beartype
from loguru import logger
from pyhere import here
from sklearn.metrics import accuracy_score
from torch import Tensor

logger.enable("amr_predict")


RNG = np.random.default_rng()


SEP: str = ";"


N: int = 1000
CHOICES = list("abcdef")


@beartype
def gen_labels(
    n: int,
    together: dict[int, tuple[list, float]] | None = None,
    reciprocate: bool = True,
) -> pl.Series:
    """
    Randomly generate label sets

    Parameters
    ----------
    together : dict[int, list] | None
        Dictionary mapping indices of CHOICES to a tuple of
        other indices which must co-occur whenever the key is seen, followed by the
        co-occurence proportion
        e.g. {1: ([0, 2], 0.8)} means that CHOICES[1] co-occurs with
        CHOICES[0] and CHOICES[2] 80% of the time

    reciprocate : bool
        If True (default behavior), the co-occuring indices also will co-occur
        with their key implicitly
        e.g. if True, {1: ([0, 2], 0.8)} means that CHOICES[1] co-occurs with
        CHOICES[0] and CHOICES[2] 80% of the time, and
        CHOICES[0] will co-occur with CHOICES[2] and CHOICES[1] 80%
        of the time

    """
    tgt = copy.deepcopy(together)
    if reciprocate and together is not None:
        for k, v in together.items():
            choices, prop = v
            for choice in choices:
                others = list(set(choices) - {choice})
                if choice not in tgt:
                    tgt[choice] = ([k] + others, prop)
                elif k not in tgt[choice][0]:
                    tgt[choice][0].extend([k] + others)

    def get():
        if not tgt:
            n_lab = RNG.integers(size=1, low=1, high=len(CHOICES), endpoint=True)
            return SEP.join(RNG.choice(CHOICES, size=n_lab, replace=False))
        chosen = RNG.integers(size=1, low=0, high=len(CHOICES)).item()
        if chosen not in tgt:
            return CHOICES[chosen]
        idx, prop = tgt[chosen]
        others = [CHOICES[i] for i in idx]
        if RNG.random(1) <= prop:
            return SEP.join([CHOICES[chosen]] + others)
        return CHOICES[chosen]

    return pl.Series([get() for _ in range(n)])


@define
class DummyLatent:
    threshold: float
    labs: list[str] | None
    fire_prop: float = 1.0
    fnr: float = 0.0
    fpr: float = 0.0

    def activation(self, labels: pl.Series) -> Tensor:
        n = len(labels)
        dead = torch.distributions.Uniform(0, self.threshold).sample((n,))
        active = torch.distributions.Uniform(self.threshold, 1).sample((n,))
        if self.labs is not None:
            has_label = labels.str.split(SEP).list.set_intersection(
                self.labs
            ).list.len() == len(self.labs)
            if self.fnr > 0:
                mask_out = RNG.choice(has_label.arg_true(), size=int(self.fnr * n))
                has_label[mask_out] = False
            if self.fpr > 0:
                mask_in = RNG.choice((~has_label).arg_true(), size=int(self.fpr * n))
                has_label[mask_in] = True
        else:
            has_label = RNG.choice([True, False], replace=True, size=n)
        result = torch.where(torch.tensor(has_label), active, dead)
        if self.fire_prop < 1:
            mask_out = RNG.choice(range(n), size=int(self.fire_prop * n))
            result[mask_out] = dead[mask_out]
        return result


def tester(
    true_firing: dict[str, list[str]],
    label_args: dict | None = None,
    threshold: float = 0.5,
    fire_prop=1.0,
    fpr=0.0,
    fnr=0.0,
):
    label_args = label_args or {}
    latents = {
        key: DummyLatent(
            threshold=threshold, labs=v, fire_prop=fire_prop, fpr=fpr, fnr=fnr
        )
        for key, v in true_firing.items()
    }
    try:
        k = max(map(len, true_firing.values()))
    except TypeError:
        k = 2
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
    template = {"by": [], "latent_idx": [], "top_label_acc": [], "truth_labels": []}
    reports = []
    for m in metrics:
        report = metric_obj.report(k=k + 1, by=m)
        to_append = (
            report.with_columns(pl.lit(m).alias("by"), cs.list().list.join(","))
            .select(["latent_idx", "by", "label", m])
            .rename({m: "value"})
        )
        reports.append(to_append)
        for latent, truth in true_firing.items():
            length = len(truth) if truth is not None else 2
            top_labels = report.filter(pl.col("latent_idx") == latent)["label"][0]
            top_labels = sorted(top_labels.to_list()[:length])
            if truth is None:
                acc = np.nan
                template["truth_labels"].append("")
            else:
                truth = sorted(truth)
                acc = accuracy_score(y_true=truth, y_pred=top_labels)
                template["truth_labels"].append(",".join(truth))
            template["by"].append(m)
            template["latent_idx"].append(latent)
            template["top_label_acc"].append(acc)
    metric_acc = pl.DataFrame(template)
    result = pl.concat(reports, how="diagonal_relaxed").join(
        metric_acc,
        on=["latent_idx", "by"],
    )
    return result


# NOTE: Only normalize if the real activations > 1 and less than 0

# %%

# TODO: use this to try make a metric for polysemanticity and
# find a procedure of calling labels for latents

# Polysemantic: fires only when ALL of the labels occur together
label_params = {
    "mono": {"true_firing": {c: [c] for c in CHOICES}},
    "random": {"true_firing": {"l1": None}},
    "mono_fpr": {"true_firing": {c: [c] for c in CHOICES}, "fpr": 0.5},
    "mono_fnr": {"true_firing": {c: [c] for c in CHOICES}, "fnr": 0.5},
    "mono_fnr_fpr": {"true_firing": {c: [c] for c in CHOICES}, "fnr": 0.5, "fpr": 0.5},
    "mono_less_fp": {"true_firing": {c: [c] for c in CHOICES}, "fire_prop": 0.7},
    "mono_perfect_co-occur": {
        "label_args": {"together": {0: ([1, 2], 1.0), 3: ([4], 1.0)}},
        "true_firing": {"l1": ["a"], "l2": ["d"], "l3": ["b"]},
    },
    "mono_co-occur": {
        "label_args": {"together": {0: ([1, 2], 0.8), 3: ([4], 0.7)}},
        "true_firing": {"l1": ["a"], "l2": ["d"], "l3": ["b"]},
    },
    "poly": {
        "true_firing": {"l1": CHOICES[:2], "l2": CHOICES[3:5]},
    },
    "poly_perfect_co-occur": {
        "label_args": {"together": {0: ([1, 2], 1.0), 3: ([4], 1.0)}},
        "true_firing": {"l1": CHOICES[:2], "l2": ["a"]},
    },
    "poly_and_mono": {
        "true_firing": {
            "l1_p": ["a", "b"],
            "l1_m": ["a"],
            "l2_m": ["b"],
        },
        "label_args": {"together": {0: ([1], 0.8)}},
    },
}
wd = here("tests")
dfs = []
for k, v in label_params.items():
    df = tester(**v).with_columns(
        pl.lit(k).alias("setup"),
        pl.col("value").cast(pl.List(pl.String)).list.join(","),
    )
    dfs.append(df)
result = pl.concat(dfs)
result.write_csv(wd / "data" / "test_sae_output.csv")
