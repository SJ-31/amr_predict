#!/usr/bin/env python3

import argparse
import os
import subprocess as sp
import tomllib
from pathlib import Path
from typing import Literal


"""
Python script implementing the pipelines described in

Vladimir Smirnov, Tandy Warnow, Phylogeny Estimation Given Sequence Length Heterogeneity, Systematic Biology, Volume 70, Issue 2, March 2021, Pages 268–282, https://doi.org/10.1093/sysbio/syaa058

NOTE: To install sepp==4.5.6 with pixi, run
pixi global install --environment sepp sepp==4.5.6 --with mafft==7.526 --with pasta==1.9.3 -c bioconda -c conda-forge
"""

DEPENDENCIES: dict[str, tuple] = {
    "required": ("seqkit", "run_pasta", "run_upp", "raxml-ng")
}


def validate_cfg(file: Path) -> None:
    with open(file, "r") as f:
        cfg: dict = tomllib.load(f)
    if "threshold" in cfg:
        assert isinstance(cfg["threshold"], int)
    if "molecule" in cfg:
        assert cfg["molecule"] in {"dna", "rna", "aa"}


class Pipeline:
    def __init__(
        self, workdir: Path, prefix: str, molecule: str, threshold: int, cfg: dict
    ) -> None:
        self.threshold: int = threshold
        self.workdir: Path = workdir
        self.prefix: str = prefix
        self.molecule: str = molecule
        self.cfg: dict = cfg
        self.ran_commands: list[str] = []
        
    def run_with_checks(self, command, *to_check: Path):
        proc = sp.run(command, shell=True)
        proc.check_returncode()
        for file in to_check:
            if os.path.getsize(file) == 0:
                raise ValueError(f"Expected file `{file}` from command {command} is empty")
        self.ran_commands.append(command)

    def get_flags(self, command: str, overrides: dict) -> str:
        if command in self.cfg:
            command_dct = self.cfg[command] | overrides
            assert isinstance(command_dct, dict), (
                f"The provided configuration for {command} is not a dictionary"
            )
        else:
            command_dct = overrides
        return " ".join({f"--{k} {v}" for k, v in command_dct.items()})

    def filter_fragments(self, fasta: Path) -> tuple[Path, Path]:
        """
        Returns
        -------
        Tuple of paths to [full sequence, fragment] FASTA files
        """
        assert fasta.exists(), "FASTA file doesn't exist"
        fragments = self.workdir / "fragments.fasta"
        full = self.workdir / "full.fasta"
        self.run_with_checks(f"seqkit seq -M {self.threshold} -o {fragments} {fasta}", fragments)
        if len_max := self.cfg.get("maximum"):
            command = f"seqkit seq -m {self.threshold} {fasta} | seqkit seq -M {len_max} -o {full}"
        else:
            command = f"seqkit seq -m {self.threshold} {fasta} -o {full}"
        self.run_with_checks(command, full)
        return full, fragments

    def align_backbone_msa(self, fasta: Path) -> Path:
        """
        Returns
        -------
        Path to backbone MSA aligned with PASTA
        """
        outdir = self.workdir / "1_pasta_bb_msa"
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
        aln = outdir / "pastajob.alignment.fasta"
        self.run_with_checks(f"run_pasta {flags}", aln)
        return aln

    def raxml_wrapper(self, msa: Path, key: str, dir) -> tuple[Path, Path]:
        """
        Returns
        -------
        Tuple of paths to [tree newick, raxml model]
        """
        outdir = self.workdir / dir
        outdir.mkdir()
        defaults = {
            "prefix": outdir / "bb",
            "msa": msa,
            "msa-format": "FASTA",
            "data-type": "AA" if self.molecule == "aa" else "dna",
        }
        if "model" not in self.cfg[key]:
            defaults["model"] = "GTR+G"
        flags = self.get_flags("raxml", defaults)
        tree, model = outdir / "bb.raxml.bestTree", outdir / "bb.raxml.bestModel"
        self.run_with_checks(f"raxml-ng {flags}", tree, model)
        return tree, model

    def build_backbone_tree(self, msa: Path) -> tuple[Path, Path]:
        return self.raxml_wrapper(msa, "raxml_bb", dir="2_raxml_bb_tree")

    def align_upp(
        self, model: Path, fragments_fasta: Path, aligned: Path, tree: Path
    ) -> Path:
        outdir = self.workdir / "3_upp_alignment"
        defaults = {
            "tree": tree,
            "alignment": aligned,
            "sequence_file": fragments_fasta,
            "raxml": model,
            "molecule": self.molecule,
            "outdir": outdir,
            "output": "output",
        }
        flags = self.get_flags("upp", defaults)
        result: Path = outdir / "output_alignment.fasta"
        proc = sp.run(f"run_upp.py {flags}", shell=True)
        proc.check_returncode()
        if os.path.getsize(result) == 0:
            raise ValueError("UPP failed to produce alignment")
        return result

    def build_tree_final(self, msa: Path) -> tuple[Path, Path]:
        return self.raxml_wrapper(msa, "raxml_final", "3_raxml_final_tree")

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
    alignment = P.align_upp(
        model=raxml_pars, fragments_fasta=fragments, aligned=bb_msa, tree=bb_tree
    )
    if args["placement"]:
        P.place_fragments()
    else:
        tree, _ = P.build_tree_final(alignment)


if __name__ == "__main__":
    args = parse_args()
    main(args)
