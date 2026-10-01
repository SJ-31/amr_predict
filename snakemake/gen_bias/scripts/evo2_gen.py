#!/usr/bin/env python3
# Adapted from scripts/gene_completion.py at
import argparse
import random
from csv import DictWriter
from pathlib import Path

import numpy as np
import torch
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
        help="Prepend prompt to output sequences.",
        action=argparse.BooleanOptionalAction,
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
        "--seed",
        type=int,
        default=random.randint(0, 999999999),
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--cached-generation",
        type=bool,
        default=True,
        help="Use KV caching during generation",
    )
    parser.add_argument(
        "--batch_size",
        default=15,
        help="Number of sequences to generate in each call, to reduce memory consumption",
        action="store",
        type=int,
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
    parser.add_argument(
        "--output_scores", default=None, help="CSV file to store scores", action="store"
    )

    args = vars(parser.parse_args())
    return args


def create_batches(num: int, batch_size: int) -> list[int]:
    n_batches = num // batch_size
    remainder = num % batch_size
    if remainder != 0:
        return ([batch_size] * n_batches) + [remainder]
    return [batch_size] * n_batches


def main(args: dict):
    random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    evo_model = Evo2(args["model_name"])
    evo_model.model.to(args["device"])

    if Path(args["prompt"]).exists():
        prompt = str(SeqIO.read(args["prompt"], "fasta").seq)
    else:
        prompt = args["prompt"]

    # Object of class GenerationOutput, from vortex library
    kws = {
        "n_tokens": args["seq_len"],
        "temperature": args["temperature"],
        "top_k": args["top_k"],
        "top_p": args["top_p"],
        "cached_generation": args["cached_generation"],
        "batched": args["batched"],
        "verbose": args["verbose"],
        "force_prompt_threshold": args["force_prompt_threshold"],
    }
    if not args["batch_size"]:
        gen_output = evo_model.generate([prompt] * args["num"], **kws)
        output_seqs = gen_output.sequences
        output_scores = gen_output.logprobs_mean
    else:
        output_seqs, output_scores = [], []
        for batch in create_batches(args["num"], args["batch_size"]):
            cur_out = evo_model.generate([prompt] * batch, **kws)
            cur_seqs = cur_out
            cur_seqs = cur_out.sequences
            cur_scores = cur_out.logprobs_mean
            # Scores are the average log probability of the generated
            # sequence, obtained by softmaxing the logits
            output_seqs.extend(cur_seqs)
            output_scores.extend(cur_scores)

    if args["prepend_prompt_to_output"]:
        print("Prepending prompt...")
        output_seqs = [prompt + s for s in output_seqs if not s.startswith(prompt)]
    as_fasta = [f">{args['prefix']}{i}\n{s}" for i, s in enumerate(output_seqs)]
    with open(args["output"], "w") as f:
        f.write("\n".join(as_fasta))
    if args["output_scores"]:
        with open(args["output_scores"], "w") as f:
            writer = DictWriter(f, ["id", "score"])
            writer.writeheader()
            for i, score in enumerate(output_scores):
                writer.writerow({"id": f"{args['prefix']}{i}", "score": score})


if __name__ == "__main__":
    args = parse_args()
    main(args)
