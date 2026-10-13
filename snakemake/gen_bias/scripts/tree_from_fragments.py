#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess as sp
import tomllib
from pathlib import Path

"""
Python script implementing the pipelines described in

Vladimir Smirnov, Tandy Warnow, Phylogeny Estimation Given Sequence Length Heterogeneity, Systematic Biology, Volume 70, Issue 2, March 2021, Pages 268–282, https://doi.org/10.1093/sysbio/syaa058

NOTE: To install sepp==4.5.6 with pixi, run
pixi global install --environment sepp sepp==4.5.6 --with mafft==7.526 --with pasta==1.9.3 -c conda-forge -c bioconda

TODO: [2026-10-12 Mon] double-check that you have the newest versions of all the tools

[2026-10-12 Mon] Program versions
raxml-ng: 2.0.3
PASTA: 1.9.3
epa-ng: 0.3.8
UPP: 4.5.6
mafft: 7.526
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
    def __init__(self, cfg: dict) -> None:
        self.threshold: int = cfg["threshold"]
        self.workdir: Path = cfg["workdir"]
        self.molecule: str = cfg["molecule"]
        self.cfg: dict = cfg
        self.command_log: Path = self.workdir / "ran_commands.json"
        self.workdir.mkdir(exist_ok=True)

    def run_with_checks(self, command, *to_check: Path):
        if all(f.exists() for f in to_check):
            return
        proc = sp.run(command, shell=True)
        proc.check_returncode()
        for file in to_check:
            if os.path.getsize(file) == 0:
                raise ValueError(
                    f"Expected file `{file}` from command {command} is empty"
                )
        commands = []
        if self.command_log.exists() and os.path.getsize(self.command_log) > 0:
            with open(self.command_log, "r") as f:
                commands = json.load(f)
        commands.append(command)
        with open(self.command_log, "w") as f:
            json.dump(commands, f, indent=2)

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
        self.run_with_checks(
            f"seqkit seq -M {self.threshold} -o {fragments} {fasta}", fragments
        )
        if len_max := self.cfg.get("maximum"):
            command = f"seqkit seq -m {self.threshold} {fasta} | seqkit seq -M {len_max} -o {full}"
        else:
            command = f"seqkit seq -m {self.threshold} {fasta} -o {full}"
        self.run_with_checks(command, full)
        return full, fragments

    def align_backbone_msa(self, fasta: Path) -> Path:
        """
        WARNING: this consumes a lot of memory


        Returns
        -------
        Path to backbone MSA aligned with PASTA (version 1.9.3)
        """
        outdir = self.workdir / "1_pasta_bb_msa"
        outdir.mkdir(exist_ok=True)
        aln = outdir / "pastajob.alignment.fasta"
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
        self.run_with_checks(f"run_pasta {flags}", aln)
        return aln

    def raxml_wrapper(self, msa: Path, key: str, dir) -> tuple[Path, Path]:
        """
        Returns
        -------
        Tuple of paths to [tree newick, raxml model]
        """
        outdir = self.workdir / dir
        outdir.mkdir(exist_ok=True)
        overrides = {
            "prefix": outdir / "bb",
            "msa": msa,
            "msa-format": "FASTA",
            "data-type": "AA" if self.molecule == "aa" else "dna",
        }
        if "model" not in self.cfg.get(key, {}):
            overrides["model"] = "GTR+G"
        flags = self.get_flags("raxml", overrides)
        tree, model = outdir / "bb.raxml.bestTree", outdir / "bb.raxml.bestModel"
        self.run_with_checks(f"raxml-ng {flags}", tree, model)
        return tree, model

    def build_backbone_tree(self, msa: Path) -> tuple[Path, Path]:
        return self.raxml_wrapper(msa, "raxml_bb", dir="2_raxml_bb_tree")

    def align_upp(
        self, model: Path, fragments_fasta: Path, aligned: Path, tree: Path
    ) -> tuple[Path, Path]:
        """
        Returns
        -------
        Tuple of [alignment, alignment_masked]
        """
        outdir = self.workdir / "3_upp_realignment"
        overrides = {
            "tree": tree,
            "alignment": aligned,
            "sequence_file": fragments_fasta,
            "raxml": model,
            "molecule": self.molecule,
            "outdir": outdir,
            "output": "output",
        }
        flags = self.get_flags("upp", overrides)
        result: Path = outdir / "output_alignment.fasta"
        masked: Path = outdir / "output_alignment_masked.fasta"
        self.run_with_checks(f"run_upp {flags}", result)
        return result, masked

    def align_mafft(self, fragments_fasta: Path, aligned: Path) -> Path:
        outdir = self.workdir / "3_mafft_realignment"
        result = outdir / "realigned.fasta"
        command = (
            f"mafft --addfragments {fragments_fasta} --keeplength {aligned} > {result}"
        )
        self.run_with_checks(command, result)
        return result

    def build_tree_final(self, msa: Path) -> tuple[Path, Path]:
        return self.raxml_wrapper(msa, "raxml_final", "4_raxml_final_tree")

    def place_fragments(
        self,
        tree: Path,
        realigned: Path,
        original_aligned: Path,
        model: Path,
    ) -> Path:
        outdir = self.workdir / "4_placement"
        outdir.mkdir(exist_ok=True)
        fragments_only = outdir / "fragments_to_place.fasta"
        filter_queries = f"seqkit grep -f <(seqkit seq -n {original_aligned}) {realigned} -v > {fragments_only}"
        self.run_with_checks(filter_queries, fragments_only)
        overrides = {
            "query": fragments_only,
            "model": model,
            "tree": tree,
            "outdir": outdir,
            "msa": original_aligned,
        }
        flags = self.get_flags("epa-ng", overrides)
        result = outdir / "epa_result.jplace"
        self.run_with_checks(f"epa-ng {flags}", result)
        return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="TOML Configuration file for called programs",
        action="store",
    )
    try:
        args = vars(parser.parse_known_args())
        cfg: dict = validate_cfg(args["config"])
    except TypeError:
        cfg = {}
    parser.add_argument(
        "-i",
        "--input",
        help="FASTA file containing sequences",
        required="input" not in cfg,
        type=Path,
    )
    parser.add_argument(
        "-t",
        "--threshold",
        help="Sequences below this number will be considered fragmentary",
        action="store",
        required="threshold" not in cfg,
    )
    parser.add_argument(
        "-M",
        "--maximum",
        default=None,
        help="If provided, filter out sequences above this length, excluding them from the final tree/jplace file",
        action="store",
    )
    parser.add_argument(
        "-r",
        "--realignment_program",
        default="upp",
        help="Program to use for realigning fragments to reference MSA",
        action="store",
        choices=["upp", "mafft"],
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
        help="Output prefix. If not specified, uses the input file name",
        action="store",
    )
    parser.add_argument(
        "-w",
        "--workdir",
        required="workdir" not in cfg,
        help="Working directory for storing intermediate files",
        action="store",
        type=Path,
    )
    args = vars(parser.parse_args())
    args = cfg | args
    return args


def main(args: dict):
    P: Pipeline = Pipeline(args)
    input: Path = (
        Path(args["input"]) if isinstance(args["input"], str) else args["input"]
    )
    if not args["prefix"]:
        args["prefix"] = str(input.parent / Path(args["input"]).stem)
    full, fragments = P.filter_fragments(input)
    bb_msa = P.align_backbone_msa(full)
    bb_tree, raxml_pars = P.build_backbone_tree(bb_msa)

    if args["realignment_program"] == "upp":
        aln, aln_for_placement = P.align_upp(
            model=raxml_pars, fragments_fasta=fragments, aligned=bb_msa, tree=bb_tree
        )
    elif args["realignment_program"] == "mafft":
        aln = P.align_mafft(fragments_fasta=fragments, aligned=bb_msa)
        aln_for_placement = aln
    else:
        raise ValueError("Realignment program not recognized")

    if args["placement"]:
        placement = P.place_fragments(
            tree=bb_tree,
            original_aligned=bb_msa,
            realigned=aln_for_placement,
            model=raxml_pars,
        )
        shutil.move(placement, f"{args['prefix']}.jplace")
    else:
        tree, _ = P.build_tree_final(aln)
        shutil.move(tree, f"{args['prefix']}.nwk")


if __name__ == "__main__":
    args = parse_args()
    main(args)
