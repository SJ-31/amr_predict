#!/usr/bin/env python3
# Adapted from scripts/gene_completion.py at
import argparse
import os
import random
from csv import DictWriter
from pathlib import Path

import modal
from Bio import SeqIO

app = modal.App()
image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel")
    .pip_install("flash-attn==2.8.0.post2", "evo2")
    .env({"HF_HOME": "/huggingface"})
)
vol = modal.Volume.from_name("HF_HOME", create_if_missing=True)
hf_home = Path("/huggingface")


def create_batches(num: int, batch_size: int) -> list[int]:
    n_batches = num // batch_size
    remainder = num % batch_size
    if remainder != 0:
        return ([batch_size] * n_batches) + [remainder]
    return [batch_size] * n_batches


def parse_args(*arglist) -> dict:
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
        "--subtract_prompt_len",
        default=False,
        help="Subtract the length of the prompt from the maximum sequence length",
        action="store_true",
    )
    parser.add_argument(
        "--output_scores", default=None, help="CSV file to store scores", action="store"
    )

    args = vars(parser.parse_args(args=arglist))
    return args


@app.function(volumes={hf_home.as_posix(): vol}, image=image, gpu="L4")
def generate(prompt: str, args) -> tuple[list, list]:
    import numpy as np
    import torch
    from evo2 import Evo2

    os.environ["HF_HOME"] = str(hf_home)

    random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    evo_model = Evo2(args["model_name"])
    evo_model.model.to(args["device"])

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
    return output_seqs, output_scores


@app.local_entrypoint()
def main(*arglist):
    args = parse_args(*arglist)
    if args["seed"] is None:
        args["seed"] = random.randint(0, 999999999)
    if Path(args["prompt"]).exists():
        prompt = str(SeqIO.read(args["prompt"], "fasta").seq)
    else:
        prompt = args["prompt"]
    if args["subtract_prompt_len"]:
        args["seq_len"] = args["seq_len"] - len(prompt)

    output_seqs, output_scores = generate.remote(prompt, args)

    if args["prepend_prompt_to_output"]:
        print("Prepending prompt...")
        output_seqs = [s if s.startswith(prompt) else prompt + s for s in output_seqs]
    as_fasta = [f">{args['prefix']}{i}\n{s}" for i, s in enumerate(output_seqs)]
    with open(args["output"], "w") as f:
        f.write("\n".join(as_fasta))
    if args["output_scores"]:
        with open(args["output_scores"], "w") as f:
            writer = DictWriter(f, ["id", "score"])
            writer.writeheader()
            for i, score in enumerate(output_scores):
                writer.writerow({"id": f"{args['prefix']}{i}", "score": score})
