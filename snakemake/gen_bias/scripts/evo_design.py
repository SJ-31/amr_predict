#!/usr/bin/env python3
# Adapted from scripts/generate.py at https://github.com/evo-design/evo
import argparse
from pathlib import Path

from Bio import SeqIO
from evo import Evo, generate


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Generate sequences using the Evo model."
    )
    parser.add_argument(
        "-x",
        "--prefix",
        default="genome_ocean",
        help="Prefix to use for generated sequences",
        action="store",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="evo-1-131k-base",
        help="Evo model name",
        choices=[
            "evo-1.5-8k-base",
            "evo-1-8k-base",
            "evo-1-131k-base",
            "evo-1-8k-crispr",
            "evo-1-8k-transposon",
        ],
    )
    parser.add_argument(
        "--prompt", type=str, default="ACGT", help="Prompt for generation"
    )
    parser.add_argument(
        "--num", type=int, default=3, help="Number of sequences to sample at once"
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
        "--prepend-bos", type=bool, default=False, help="Prepend BOS token"
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
    evo_model = Evo(args["model_name"])
    model, tokenizer = evo_model.model, evo_model.tokenizer

    model.to(args["device"])
    model.eval()

    if Path(args["prompt"]).exists():
        prompt = str(next(SeqIO.parse(args["prompt"], "fasta")).seq)
    else:
        prompt = args["prompt"]

    output_seqs, output_scores = generate(
        [prompt] * args["num"],
        model,
        tokenizer,
        n_tokens=args["seq_len"],
        temperature=args["temperature"],
        top_k=args["top_k"],
        top_p=args["top_p"],
        cached_generation=args["cached_generation"],
        batched=args["batched"],
        prepend_bos=args["prepend_bos"],
        device=args["device"],
        verbose=args["verbose"],
    )
    as_fasta = [f">{args['prefix']}{i}\n{s}" for i, s in enumerate(output_seqs)]
    with open(args["output"], "w") as f:
        f.write("\n".join(as_fasta))


if __name__ == "__main__":
    args = parse_args()
    main(args)
