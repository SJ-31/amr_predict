#!/usr/bin/env python3

from collections import defaultdict
from pathlib import Path

from amr_predict.utils import EmbeddingCache, deduplicate
from datasets import Dataset, concatenate_datasets
from loguru import logger


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    args = vars(parser.parse_args())  # convert to dict
    return args


def gather_datasets(smk_dirs: list, subdir) -> dict[str, list[Path]]:
    dset_tracker = defaultdict(list)
    for smk_dir in (Path(d) for d in smk_dirs):
        current = smk_dir / subdir
        if current.exists():
            for dpath in (i for i in current.iterdir() if i.is_dir()):
                dset_name = dpath.stem
                dset_tracker[dset_name].append(dpath)
    return dset_tracker


if __name__ == "__main__":
    args = parse_args()
    dir_mapping = ("embedded", "pooled", "processed_sequences")
    out_root: Path = Path(args["output"])
    for dir in dir_mapping.items():
        name2dset_list = gather_datasets(args["input"], dir)
        cur_out: Path = out_root / dir
        cur_out.mkdir(exist_ok=True, parents=True)
        for dset, to_combine in name2dset_list.items():
            logger.info("Creating combined `{}` from paths {}", dset, to_combine)
            combined_path = cur_out / dset
            if dir == "embedded":
                new_cache = EmbeddingCache.combine(to_combine, new_path=combined_path)
            else:
                new_dset: Dataset = concatenate_datasets(
                    [Dataset.load_from_disk(d) for d in to_combine]
                )
                logger.info("Size before deuplication: {}", len(new_dset))
                new_dset = deduplicate(new_dset)
                logger.info("Size after: {}", len(new_dset))
                new_dset.save_to_disk(combined_path)
