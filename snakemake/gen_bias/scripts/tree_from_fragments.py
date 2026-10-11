#!/usr/bin/env python3

import argparse
import os
import subprocess as sp
from pathlib import Path

import tomllib

"""
Python script implementing the pipelines described in

Vladimir Smirnov, Tandy Warnow, Phylogeny Estimation Given Sequence Length Heterogeneity, Systematic Biology, Volume 70, Issue 2, March 2021, Pages 268–282, https://doi.org/10.1093/sysbio/syaa058
"""

DEPENDENCIES: dict[str, tuple] = {
    "required": ("seqkit", "run_pasta", "run_upp", "raxml-ng")
}
CONFIG_OVERRIDES: tuple[str] = ("threshold", "maximum", "input")


def validate_cfg(file: Path) -> None:
    with open(file, "r") as f:
        cfg: dict = tomllib.load(f)
    if "threshold" in cfg:
        assert isinstance(cfg["threshold"], int)
    if "molecule" in cfg:
        assert cfg["molecule"] in {"dna", "rna", "aa"}


class Pipeline:
    def __init__(self, workdir: Path, prefix: str, molecule: str, cfg: dict) -> None:
        self.workdir: Path = workdir
        self.prefix: str = prefix
        self.molecule: str = molecule
        self.cfg: dict = cfg
        self.ran_commands: list[str] = []

    def get_flags(self, command: str, overrides: dict) -> str:
        if command in self.cfg:
            command_dct = self.cfg[command] | overrides
            assert isinstance(
                command_dct, dict
            ), f"The provided configuration for {command} is not a dictionary"
        else:
            command_dct = overrides
        return " ".join({f"--{k} {v}" for k, v in command_dct.items()})

    def filter_fragments(self, file: Path) -> tuple[Path, Path]:
        """
        Returns
        -------
        Tuple of paths to [full sequence, fragment] FASTA files
        """
        assert file.exists(), "FASTA file doesn't exist"
        pass

    def align_backbone_msa(self, fasta: Path) -> Path:
        """
        Returns
        -------
        Path to backbone MSA aligned with PASTA
        """
        outdir = self.workdir / "pasta_bb_msa"
        outdir.mkdir()
        flags = self.get_flags(
            "pasta",
            {
                "temporaries": outdir / ".tmp",
                "output-directory": outdir,
                "input": fasta,
                "datatype": self.molecule,
                "alignment-suffix": "alignment.fasta",
            },
        )
        proc = sp.run(f"run_pasta {flags}", shell=True)
        self.ran_commands.append(f"run_pasta {flags}")
        proc.check_returncode()
        aln = outdir / "pastajob.alignment.fasta"
        if os.path.getsize(aln) == 0:
            raise ValueError("PASTA failed to produce alignment")
        return aln

    def build_backbone_tree(self, msa: Path) -> tuple[Path, Path]:
        """
        Returns
        -------
        Tuple of paths to [tree newick, raxml model]
        """
        outdir = self.workdir / "raxml_bb_tree"
        outdir.mkdir()
        defaults = {
            "prefix": outdir / "bb",
            "msa": msa,
            "msa-format": "FASTA",
            "data-type": "AA" if self.molecule == "aa" else "dna",
        }
        if "model" not in self.cfg["raxml"]:
            defaults["model"] = "GTR+G"
        flags = self.get_flags("raxml", defaults)
        proc = sp.run(f"raxml-ng {flags}", shell=True)
        self.ran_commands.append(f"raxml-ng {flags}")
        proc.check_returncode()
        tree = outdir / "bb.raxml.bestTree"
        model = outdir / "bb.raxml.bestModel"
        if os.path.getsize(tree) == 0:
            raise ValueError("raxml-ng failed to produce tree")
        if os.path.getsize(model) == 0:
            raise ValueError("raxml-ng failed to produce model file")
        return tree, model

    def align_upp(self) -> Path:
        pass

    def build_tree_final(self) -> Path:
        pass

    def place_fragments(self) -> Path:
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="TOML Configuration file for called programs",
        action="store",
    )
    args = vars(parser.parse_args())
    if cfg_file := args.get("config"):
        cfg: dict = validate_cfg(cfg_file)
    else:
        cfg = {}
    parser.add_argument(
        "-i",
        "--input",
        help="FASTA file containing sequences",
        required="input" in cfg,
    )
    parser.add_argument(
        "-t",
        "--threshold",
        help="Sequences below this number will be considered fragmentary",
        action="store",
        required="threshold" in cfg,
    )
    parser.add_argument(
        "-M",
        "--maximum",
        default=None,
        help="If provided, filter out sequences above this length, excluding them from the final tree/jplace file",
        action="store",
    )
    parser.add_argument(
        "-u",
        "--molecule",
        default="dna",
        help="Specify type of molecule",
        action="store",
        choices=["dna", "rna", "aa"],
    )
    parser.add_argument(
        "-p",
        "--placement",
        default=False,
        help="""
        Use placement instead of building the tree denovo.
        If this flag is enabled, then the final output will be a jplace file, not a newick tree.
        """,
        action="store_true",
    )
    parser.add_argument(
        "-x",
        "--prefix",
        default=None,
        help="Output prefix. If not specified, use the input file name",
        action="store",
    )
    parser.add_argument(
        "-o",
        "--output",
        required="output" in cfg,
        help="Working directory for storing intermediate files",
        action="store",
        type=Path,
    )
    args = vars(parser.parse_args())
    args = cfg | args
    return args


def main(args: dict):
    P: Pipeline = Pipeline(args)
    full, fragments = P.filter_fragments(args["input"])
    bb_msa = P.align_backbone_msa(full)
    bb_tree, raxml_pars = P.build_backbone_tree()
    P.align_upp()
    if args["placement"]:
        P.place_fragments()
    else:
        P.build_tree_final()


if __name__ == "__main__":
    args = parse_args()
    main(args)
