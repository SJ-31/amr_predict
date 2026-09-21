# Adapted from genomeocean/go_generate.py
# https://github.com/jgi-genomeocean/genomeocean/blob/main/go_generate.py

import argparse
import os
import random

from genomeocean.generation import SequenceGenerator

if "VLLM_USE_V1" in os.environ:
    del os.environ["VLLM_USE_V1"]


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Generate DNA sequences using GenomeOcean model."
    )

    # Create a mutually exclusive group for model_dir and model
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model_dir",
        type=str,
        help="Model directory or path to a local copy of the model.",
    )
    model_group.add_argument(
        "--model",
        type=str,
        choices=["100M", "500M", "4B"],
        help="Predefined model to use.",
        default="100M",
    )

    parser.add_argument(
        "--prompt", type=str, help="File containing DNA sequences as prompts."
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Number of sequences to generate for each prompt.",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        help="Minimum length of generated sequences in tokens.",
        default=512,
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=10240,
        help="Maximum length of generated sequences in tokens.",
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature for sampling.", default=0.7
    )
    parser.add_argument("--top_k", type=int, default=-1, help="Top_k for sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top_p for sampling.")
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0,
        help="Presence penalty for sampling.",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0,
        help="Frequency penalty for sampling.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        help="Repetition penalty for sampling.",
        default=1.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(0, 999999999),
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--prepend_prompt_to_output",
        type=bool,
        default=True,
        help="Prepend prompt to output sequences.",
    )
    parser.add_argument(
        "--max_repeats",
        type=float,
        default=100,
        help="Remove sequences with more than k% simple repeats.",
    )

    # Generate a default output prefix with a randomized number
    default_out_prefix = os.path.join(
        os.getcwd(), f"go_seq_{random.randint(1000, 9999)}"
    )
    parser.add_argument(
        "--out_prefix", type=str, default=default_out_prefix, help="Output file prefix."
    )

    parser.add_argument(
        "--out_format",
        type=str,
        choices=["fa", "txt"],
        default="fa",
        help="Output format (txt or fa).",
    )

    parser.add_argument(
        "--preset",
        type=str,
        choices=["conservative", "conservative_long", "creative", "creative_long"],
        default="conservative",
        help="Preset configuration for generation parameters.",
    )

    args: dict = vars(parser.parse_args())

    # Determine the model directory based on the provided arguments
    if args.get("model") == "4B":
        args["model_dir"] = "pGenomeOcean/GenomeOcean-4B"
    elif args.get("model") == "100M":
        args["model_dir"] = "pGenomeOcean/GenomeOcean-100M"
    elif args.get("model") == "500M":
        args["model_dir"] = "pGenomeOcean/GenomeOcean-500M"
    if not args.get("model_dir"):
        args["model_dir"] = "pGenomeOcean/GenomeOcean-100M"

    return args


def main(args: dict):
    # Initialize the SequenceGenerator with the provided arguments
    seq_gen = SequenceGenerator(
        model_dir=args["model_dir"],
        promptfile=args["prompt"],
        num=args["num"],
        min_seq_len=args["min_seq_len"],
        max_seq_len=args["seq_len"],
        temperature=args["temperature"],
        top_k=args["top_k"],
        top_p=args["top_p"],
        presence_penalty=args["presence_penalty"],
        frequency_penalty=args["frequency_penalty"],
        repetition_penalty=args["repetition_penalty"],
        seed=args["seed"],
    )

    # Generate sequences
    all_generated = seq_gen.generate_sequences(
        prepend_prompt_to_output=args["prepend_prompt_to_output"],
        max_repeats=args["max_repeats"],
    )

    # Save the generated sequences to the specified output file
    seq_gen.save_sequences(
        all_generated,
        out_prefix=args["out_prefix"],
        out_format=args["out_format"],
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
