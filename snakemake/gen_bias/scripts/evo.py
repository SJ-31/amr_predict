#!/usr/bin/env python3
# Adapted from scripts/gene_completion.py at
import argparse
from pathlib import Path

from Bio import SeqIO
from evo2 import Evo2


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Generate sequences using the Evo model."
    )
    parser.add_argument(
        "-x",
        "--prefix",
        default="evo2_",
        help="Prefix to use for generated sequences",
        action="store",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="evo2_7b",
        help="Evo model name",
        choices=["evo2_7b", "evo2_7b_262k", "evo2_7b_base"],
    )
    parser.add_argument(
        "--prompt", type=str, default="ACGT", help="Prompt for generation"
    )
    parser.add_argument(
        "--num", type=int, default=3, help="Number of sequences to sample at once"
    )
    parser.add_argument(
        "--prepend_prompt_to_output",
        type=bool,
        default=True,
        help="Prepend prompt to output sequences.",
    )
    parser.add_argument(
        "-t",
        "--force_prompt_threshold",
        default=None,
        help="""
        If specified, avoids OOM errors through teacher forcing if the prompt is longer than this threshold.

        If force_prompt_threshold is none, sets default assuming 1xH100 (evo2_7b) and 2xH100 (evo2_40b) to help avoid OOM errors.
        """,
        action="store",
    )
    parser.add_argument(
        "--seq_len", type=int, default=100, help="Maximum sequence length"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature during sampling"
    )
    parser.add_argument("--top_k", type=int, default=4, help="Top K during sampling")
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="Top P during sampling"
    )
    parser.add_argument(
        "--cached-generation",
        type=bool,
        default=True,
        help="Use KV caching during generation",
    )
    parser.add_argument(
        "--batched", type=bool, default=True, help="Use batched generation"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device for generation"
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument(
        "-o", "--output", required=True, help="Output file", action="store"
    )

    args = vars(parser.parse_args())
    return args


def main(args: dict):
    evo_model = Evo2(args["model_name"])
    evo_model.model.to(args["device"])

    if Path(args["prompt"]).exists():
        prompt = str(SeqIO.read(args["prompt"], "fasta").seq)
    else:
        prompt = args["prompt"]

    output_seqs, output_scores = evo_model.generate(
        [prompt] * args["num"],
        n_tokens=args["seq_len"],
        temperature=args["temperature"],
        top_k=args["top_k"],
        top_p=args["top_p"],
        cached_generation=args["cached_generation"],
        batched=args["batched"],
        verbose=args["verbose"],
        force_prompt_threshold=args["force_prompt_threshold"],
    )
    if args["prepend_prompt_to_output"]:
        output_seqs = [s + prompt for s in output_seqs if not s.startswith(prompt)]
    as_fasta = [f">{args['prefix']}{i}\n{s}" for i, s in enumerate(output_seqs)]
    with open(args["output"], "w") as f:
        f.write("\n".join(as_fasta))


if __name__ == "__main__":
    args = parse_args()
    main(args)
