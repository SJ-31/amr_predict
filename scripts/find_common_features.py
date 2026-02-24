#!/usr/bin/env python3

import polars as pl


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to seq_metadata.csv")
    parser.add_argument(
        "-f",
        "--feature_cols",
        help="Columns in seq_metadata.csv to use as features",
        nargs="+",
        action="extend",
    )
    parser.add_argument(
        "-s",
        "--sample_col",
        default="sample",
        help="Sample identifier column",
        action="store",
    )
    parser.add_argument(
        "-c",
        "--count",
        default=5000,
        help="Number of common features to keep. If more common features than request are present, will select the most variable ones",
        action="store",
        type=int,
    )
    parser.add_argument("-o", "--output")
    parser.add_argument("-t", "--test", default=False, help="Test", action="store_true")
    args = vars(parser.parse_args())  # convert to dict
    return args


if __name__ == "__main__":
    args = parse_args()
    df: pl.DataFrame = pl.read_csv(args["input"])
    fcols = args["feature_cols"]
    scol = args["sample_col"]
    result = (
        df.select([scol] + fcols)
        .unpivot(index=scol, value_name="var")
        .drop("variable")
        .group_by("var")
        .agg(
            pl.col(scol).len().alias("n"),
            pl.col(scol).unique().len().alias("n_unique"),
        )
        .sort("n_unique", descending=True)
        .filter(pl.col("var") != "null")
        .head(args["count"])["var"]
    )
    with open(args["output"], "w") as f:
        f.write("\n".join(result))
