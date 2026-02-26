#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path

import polars as pl
from amr_predict.utils import EmbeddingCache, deduplicate
from datasets import Dataset, concatenate_datasets, disable_progress_bar
from datashader.composite import source
from loguru import logger

logger.disable("amr_predict")
disable_progress_bar()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", nargs="+", action="extend")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument(
        "-f",
        "--funcscan",
        default=False,
        help="Combine funcscan runs instead of pipeline output",
        action="store_true",
    )
    args = vars(parser.parse_args())  # convert to dict
    return args


def gather_datasets(smk_dirs: list, subdir) -> dict[str, list[Path]]:
    dset_tracker = defaultdict(list)
    for smk_dir in (Path(d) for d in smk_dirs):
        current = smk_dir / "datasets" / subdir
        if current.exists():
            for dpath in (i for i in current.iterdir() if i.is_dir()):
                dset_name = dpath.stem
                dset_tracker[dset_name].append(dpath)
        else:
            logger.info(f"WARNING: expected path {current} doesn't exist")
    return dset_tracker


if __name__ == "__main__":
    args = parse_args()
    out_root: Path = Path(args["output"])
    if not args["funcscan"]:
        dir_mapping = ("embedded", "pooled", "processed_sequences")
        for dir in dir_mapping:
            name2dset_list = gather_datasets(args["input"], dir)
            cur_out: Path = out_root / dir
            cur_out.mkdir(exist_ok=True, parents=True)
            for dset, to_combine in name2dset_list.items():
                logger.info("Creating combined `{}` from paths {}", dset, to_combine)
                combined_path = cur_out / dset
                if combined_path.exists():
                    print(f"Target path {combined_path} exists already, ignoring...")
                    continue
                if dir == "embedded":
                    new_cache = EmbeddingCache.combine(
                        to_combine, new_path=combined_path
                    )
                else:
                    new_dset: Dataset = concatenate_datasets(
                        [Dataset.load_from_disk(d) for d in to_combine]
                    )
                    logger.info("Size before deduplication: {}", len(new_dset))
                    key = "sample" if dir == "pooled" else "uid"
                    new_dset = deduplicate(new_dset, key=key)
                    logger.info("Size after: {}", len(new_dset))
                    new_dset.save_to_disk(combined_path)
    else:
        outputs = {
            "hamr": out_root / "hamronization_combined_report.tsv",
            "ampcombi": out_root / "Ampcombi_summary_cluster.tsv",
            "combgc": out_root / "combgc",
            "bakta": out_root / "bakta",
        }
        tsv_to_combine = defaultdict(list)
        dirs_to_merge: dict[str, list[Path]] = defaultdict(list)
        for funcscan_output in (Path(p) for p in args["input"]):
            hamr = (
                funcscan_output
                / "reports/hamronization_summarize/hamronization_combined_report.tsv"
            )
            if hamr.exists():
                tsv_to_combine["hamr"].append(hamr)
            else:
                print(f"WARNING: no HAMR file for {funcscan_output}")
            ampcombi_1 = (
                funcscan_output / "reports/ampcombi2/Ampcombi_summary_cluster.tsv"
            )
            if not ampcombi_1.exists():
                ampcombi_2 = funcscan_output / "reports/ampcombi2/Ampcombi_summary.tsv"
                if not ampcombi_2.exists():
                    print(f"WARNING: no Ampcombi file for {funcscan_output}")
                else:
                    tsv_to_combine["ampcombi"].append(ampcombi_2)
            else:
                tsv_to_combine["ampcombi"].append(ampcombi_1)
            for n, dir in zip(
                ("bakta", "combgc"), ("annotation/bakta/all", "reports/combgc")
            ):
                to_check = funcscan_output / dir
                if not to_check.exists():
                    print(f"WARNING: no {n} dir exists for {funcscan_output}")
                else:
                    dirs_to_merge[n].append(to_check)
        for name, tsvs in tsv_to_combine.items():
            merged = pl.concat(
                [pl.read_csv(t, separator="\t", null_values="NA") for t in tsvs],
                how="diagonal_relaxed",
            )
            merged.write_csv(outputs[name], separator="\t")
        for name, sources in dirs_to_merge.items():
            outdir = outputs[name]
            outdir.mkdir(exist_ok=True)
            for source in sources:
                for file in source.iterdir():
                    try:
                        (outdir / file.name).symlink_to(file)
                    except FileExistsError:
                        continue
