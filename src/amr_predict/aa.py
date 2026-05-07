#!/usr/bin/env python3

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from functools import reduce
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
from scipy import stats
from skbio.sequence import SubstitutionMatrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def convert_property_table(table: str, converter: Callable | None = None) -> dict:
    result = {}
    lines = table.strip().split("\n")
    assert len(lines) == 3
    index2aa: dict = {}
    result: dict = {}
    for i, line in enumerate(lines):
        splits = list(filter(lambda x: x, map(lambda x: x.strip(), line.split(" "))))
        for j, item in enumerate(splits):
            if i == 0:
                aa_split = item.split("/")
                index2aa[(i + 1, j)] = aa_split[0]
                index2aa[(i + 2, j)] = aa_split[1]
            else:
                lookup = index2aa[(i, j)]
                result[lookup] = converter(item) if converter else item
    return result


aa_tables: dict[str, str] = {
    # average normalized https://www.genome.jp/entry/aaindex:VINM940101
    "flexibility": """
 A/L     R/K     N/M     D/F     C/P     Q/S     E/T     G/W     H/Y     I/V
   0.984   1.008   1.048   1.068   0.906   1.037   1.094   1.031   0.950   0.927
   0.935   1.102   0.952   0.915   1.049   1.046   0.997   0.904   0.929   0.931
    """,
    # https://www.genome.jp/entry/aaindex:JURD980101
    "hydrophobicity": """
 A/L     R/K     N/M     D/F     C/P     Q/S     E/T     G/W     H/Y     I/V
    1.10   -5.10   -3.50   -3.60    2.50   -3.68   -3.20   -0.64   -3.20    4.50
    3.80   -4.11    1.90    2.80   -1.90   -0.50   -0.70   -0.46    -1.3     4.2
    """,
    # https://www.genome.jp/entry/aaindex:FASG760101
    "mw": """
  A/L     R/K     N/M     D/F     C/P     Q/S     E/T     G/W     H/Y     I/V
   89.09  174.20  132.12  133.10  121.15  146.15  147.13   75.07  155.16  131.17
  131.17  146.19  149.21  165.19  115.13  105.09  119.12  204.24  181.19  117.15
""",
    "charge": """
 A/L     R/K     N/M     D/F     C/P     Q/S     E/T     G/W     H/Y     I/V
      0.      1.      0.     -1.      0.      0.     -1.      0.      0.      0.
      0.      1.      0.      0.      0.      0.      0.      0.      0.      0.
""",
    # https://www.genome.jp/entry/aaindex:KRIW790103
    "side_chain_volume": """
A/L     R/K     N/M     D/F     C/P     Q/S     E/T     G/W     H/Y     I/V
    27.5   105.0    58.7    40.0    44.6    80.7    62.0     0.0    79.0    93.5
    93.5   100.0    94.1   115.5    41.9    29.3    51.3   145.5   117.3    71.5
""",
    # https://www.genome.jp/entry/aaindex:ZIMJ680104
    "isoelectric_point": """
 A/L     R/K     N/M     D/F     C/P     Q/S     E/T     G/W     H/Y     I/V
    6.00   10.76    5.41    2.77    5.05    5.65    3.22    5.97    7.59    6.02
    5.98    9.74    5.74    5.48    6.30    5.68    5.66    5.89    5.66    5.96
""",
}


def safe_float(x: str) -> float:
    return float(x) if x != "NA" else np.nan


other_conversion = {"charge": lambda x: int(x.removesuffix("."))}

AA_PROPERTIES: dict[str, dict] = {
    name: convert_property_table(tab, other_conversion.get(name, safe_float))
    for name, tab in aa_tables.items()
}


def make_aa_df(cluster_kws: dict | None = None) -> pl.DataFrame:
    df: pl.DataFrame = reduce(
        lambda x, y: x.join(y, on="aa"),
        [
            pl.DataFrame({"aa": v.keys(), k: v.values()})
            for k, v in AA_PROPERTIES.items()
        ],
    )
    # Normalize properties so that those on different scales don't dominate
    # in clustering
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.drop("aa"))

    clst = AgglomerativeClustering(**(cluster_kws or {}))
    clusters = clst.fit_predict(scaled)
    df = df.with_columns(pl.Series(clusters).alias("cluster"))
    return df


AA_DF = make_aa_df(cluster_kws={"n_clusters": 10})


# Check groups with substitution matrix scoring
def check_groups():
    groups: list[list] = AA_DF.group_by("cluster").agg("aa")["aa"].to_list()
    scores = defaultdict(list)
    for name in ("BLOSUM90", "BLOSUM50", "PAM250", "PAM70"):
        mat = SubstitutionMatrix.by_name(name)
        array = mat.to_data_frame().values
        avg = array.mean()
        median = np.median(array)
        iqr = stats.iqr(array)
        group_vals = []
        for group in groups:
            vals = []
            for aa1 in group:
                for aa2 in group:
                    if aa1 == aa2:
                        continue
                    vals.append(mat[aa1, aa2])
            group_vals.append(list(set(vals)))
        scores["matrix"].append(name)
        scores["max"].append(array.max())
        scores["min"].append(array.min())
        scores["mean"].append(avg)
        scores["iqr"].append(iqr)
        scores["median"].append(median)
        scores["group"].append(groups)
        scores["group_values"].append(group_vals)
    return (
        pl.DataFrame(scores)
        .explode("group", "group_values")
        .cast({"group_values": pl.List(pl.Utf8)})
        .with_columns(cs.list().list.join(";"))
    )


# [2026-05-07 Thu] as expected the groups don't exactly correspond to
# the highest scoring pairs in the substitution matrices


def read_paml(x: str) -> SubstitutionMatrix:
    """
    Read a substitution matrix stored in a PAML file or string

    https://github.com/iqtree/iqtree3/blob/2e2b6bcea0c6cdbbe487e768488ebcfb8d4bc62b/model/modelprotein.cpp#L851
    """
    if "\n" not in x and Path(x).exists():
        x = Path(x).read_text()
    lines = x.strip().split("\n")
    result = defaultdict(dict)
    aa_order = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]
    for i, line in enumerate(lines):
        splits = line.strip().split(" ")
        for j, val in enumerate(splits):
            result[aa_order[j]][aa_order[i + 1]] = float(val)
            result[aa_order[i + 1]][aa_order[j]] = float(val)
    for k, v in result.items():
        v[k] = np.nan
    return SubstitutionMatrix.from_dict(result)


PHYLO_MATRICES: dict[str, str] = {
    "LG": """
0.425093
0.276818 0.751878
0.395144 0.123954 5.076149
2.489084 0.534551 0.528768 0.062556
0.969894 2.807908 1.695752 0.523386 0.084808
1.038545 0.363970 0.541712 5.243870 0.003499 4.128591
2.066040 0.390192 1.437645 0.844926 0.569265 0.267959 0.348847
0.358858 2.426601 4.509238 0.927114 0.640543 4.813505 0.423881 0.311484
0.149830 0.126991 0.191503 0.010690 0.320627 0.072854 0.044265 0.008705 0.108882
0.395337 0.301848 0.068427 0.015076 0.594007 0.582457 0.069673 0.044261 0.366317 4.145067
0.536518 6.326067 2.145078 0.282959 0.013266 3.234294 1.807177 0.296636 0.697264 0.159069 0.137500
1.124035 0.484133 0.371004 0.025548 0.893680 1.672569 0.173735 0.139538 0.442472 4.273607 6.312358 0.656604
0.253701 0.052722 0.089525 0.017416 1.105251 0.035855 0.018811 0.089586 0.682139 1.112727 2.592692 0.023918 1.798853
1.177651 0.332533 0.161787 0.394456 0.075382 0.624294 0.419409 0.196961 0.508851 0.078281 0.249060 0.390322 0.099849 0.094464
4.727182 0.858151 4.008358 1.240275 2.784478 1.223828 0.611973 1.739990 0.990012 0.064105 0.182287 0.748683 0.346960 0.361819 1.338132
2.139501 0.578987 2.000679 0.425860 1.143480 1.080136 0.604545 0.129836 0.584262 1.033739 0.302936 1.136863 2.020366 0.165001 0.571468 6.472279
0.180717 0.593607 0.045376 0.029890 0.670128 0.236199 0.077852 0.268491 0.597054 0.111660 0.619632 0.049906 0.696175 2.457121 0.095131 0.248862 0.140825
0.218959 0.314440 0.612025 0.135107 1.165532 0.257336 0.120037 0.054679 5.306834 0.232523 0.299648 0.131932 0.481306 7.803902 0.089613 0.400547 0.245841 3.151815
2.547870 0.170887 0.083688 0.037967 1.959291 0.210332 0.245034 0.076701 0.119013 10.649107 1.702745 0.185202 1.898718 0.654683 0.296501 0.098369 2.188158 0.189510 0.249313
    """,
    "Q_PFAM": """
0.531344742
0.266631781 0.610524242
0.479415354 0.145193836 4.395589145
2.490407258 0.797726764 0.617331366 0.086320436
1.058818226 2.850794598 1.685541958 0.623180282 0.163023963
1.178483844 0.358512949 0.572153867 4.775857514 0.004224284 4.045465925
1.897932882 0.427923043 1.417171473 0.993358642 0.723368327 0.349584077 0.412573692
0.453464468 2.765967911 4.395995003 0.944775779 1.220289602 4.992256584 0.444851397 0.432227777
0.176454495 0.103023046 0.192557924 0.012280201 0.599859067 0.090487083 0.045755066 0.025135568 0.108027826
0.419433650 0.307712278 0.070917051 0.019106538 0.827369996 0.609556427 0.066812844 0.070095729 0.420152907 4.316039810
0.501174376 5.070603955 2.126974783 0.311739636 0.042113153 3.211891588 1.628511729 0.323774881 0.779002069 0.188076678 0.128912693
1.175077280 0.565654138 0.405987508 0.044371788 1.330027314 1.704580053 0.217689890 0.196370346 0.488138895 5.052397990 6.674964742 0.790881216
0.266730243 0.050284344 0.098902029 0.023281590 1.570975979 0.044860498 0.016021778 0.116848629 0.754840320 1.333037626 2.539936322 0.022355802 1.944915128
1.371519672 0.396818483 0.230865860 0.538351193 0.117103191 0.764761023 0.532587532 0.323334201 0.635697033 0.101194285 0.285369684 0.446883087 0.146079999 0.106999973
3.745215188 0.711941869 3.559567653 1.188991720 3.084090802 1.178336151 0.587105765 1.645856748 1.047872072 0.078854093 0.182164122 0.756113129 0.449949835 0.319772739 1.499944856
2.134156546 0.664457704 1.923647217 0.498902533 1.382576296 1.172710577 0.733559201 0.232722239 0.648977470 1.067438502 0.303042511 1.169845557 2.176841262 0.185666747 0.667935113 5.538833003
0.198118657 0.503943335 0.048383536 0.030638664 0.964172901 0.185024339 0.044029570 0.199917400 0.592439554 0.143667666 0.614180565 0.040725071 0.765641182 2.174974075 0.133590865 0.217347672 0.125958817
0.208747809 0.263834003 0.570488409 0.134714779 1.863419357 0.264729772 0.100447410 0.074554161 5.545635324 0.271724216 0.338670344 0.138599247 0.651180870 7.474120415 0.108442089 0.374198514 0.267599595 2.604411280
2.688525915 0.201107356 0.119873351 0.052396485 2.371412865 0.282178057 0.297627071 0.134209258 0.229732340 11.786948184 2.030164484 0.222793132 2.397008325 0.758096789 0.362295352 0.127446562 2.284500453 0.201953893 0.337688120
    """,
}
