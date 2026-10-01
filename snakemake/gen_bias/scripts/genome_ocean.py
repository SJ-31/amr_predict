# Adapted from genomeocean/go_generate.py
# https://github.com/jgi-genomeocean/genomeocean/blob/main/go_generate.py

import argparse
import os
import random
from csv import DictWriter
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from genomeocean.generation import SequenceGenerator
from transformers import PreTrainedTokenizerFast
from vllm import LLM, SamplingParams

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
        "--model_name",
        type=str,
        choices=["GenomeOcean-4B", "GenomeOcean-100M", "GenomeOcean-500M"],
        help="Predefined model to use.",
        default="GenomeOcean-100M",
    )
    parser.add_argument(
        "-x",
        "--prefix",
        default="genome_ocean",
        help="Prefix to use for generated sequences",
        action="store",
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
        default=None,
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
        help="Prepend prompt to output sequences.",
        action=argparse.BooleanOptionalAction,
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
        "--output", type=str, default=default_out_prefix, help="Output file prefix."
    )

    parser.add_argument(
        "--preset",
        type=str,
        choices=["conservative", "conservative_long", "creative", "creative_long"],
        default="conservative",
        help="Preset configuration for generation parameters.",
    )
    parser.add_argument(
        "--batch_size",
        default=15,
        help="Number of sequences to generate in each call, to reduce memory consumption",
        action="store",
        type=int,
    )
    parser.add_argument(
        "--output_scores", default=None, help="CSV file to store scores", action="store"
    )

    args: dict = vars(parser.parse_args())
    if not args["min_seq_len"]:
        args["min_seq_len"] = 10
    model_name = f"pGenomeOcean/{args['model_name']}"
    if not args.get("model_dir"):
        args["model_dir"] = model_name

    return args


def create_batches(num: int, batch_size: int) -> list[int]:
    n_batches = num // batch_size
    remainder = num % batch_size
    if remainder != 0:
        return ([batch_size] * n_batches) + [remainder]
    return [batch_size] * n_batches


def generate(seq_gen, num: int | None = None) -> tuple[list[str], list[float]]:
    """
    Custom generation method that records scores

    Modified from https://github.com/jgi-genomeocean/genomeocean/genomeocean/llm_utils.py
    """
    prompts = seq_gen._load_prompts()
    llm_utils = seq_gen.llm

    if llm_utils._vllm_engine is None:
        if llm_utils._model is not None:
            print("Unloading Transformers model to load vLLM engine...")
            del llm_utils._model
            llm_utils._model = None
            import gc

            gc.collect()
            torch.cuda.empty_cache()

        llm_utils._vllm_engine = LLM(
            model=llm_utils.model_dir,
            trust_remote_code=False,
            seed=seq_gen.seed,
            dtype=torch.bfloat16,
            max_model_len=llm_utils.model_max_length,
            gpu_memory_utilization=llm_utils.gpu_memory_utilization,
            enforce_eager=True,
            tensor_parallel_size=llm_utils.gpus,
            skip_tokenizer_init=True,
        )
    llm_utils._vllm_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        llm_utils.model_dir
    )

    llm = llm_utils._vllm_engine
    tokenizer = llm_utils._vllm_tokenizer
    prompts = ["[CLS]" + p for p in prompts]

    prompt_inputs = [
        {"prompt_token_ids": tokenizer.encode(p, add_special_tokens=False)}
        for p in prompts
    ]

    sampling_params = SamplingParams(
        n=num or seq_gen.num,
        temperature=seq_gen.temperature,
        top_k=seq_gen.top_k,
        top_p=seq_gen.top_p,
        stop_token_ids=[2],
        max_tokens=seq_gen.max_seq_len,
        min_tokens=seq_gen.min_seq_len,
        detokenize=False,
        presence_penalty=seq_gen.presence_penalty,
        frequency_penalty=seq_gen.frequency_penalty,
        logprobs=1,  # [2026-10-01 Thu] apparently the max is 20
        # Only keeps the highest twenty token probabilities
        repetition_penalty=seq_gen.repetition_penalty,
        logit_bias={8: float("-inf")},
        # Block token 8 ('N'); replaces V0 allowed_token_ids
    )

    # Generate sequences using prompt_token_ids
    generated_sequences, scores = [], []
    all_outputs = llm.generate(
        prompts=prompt_inputs,
        sampling_params=sampling_params,
    )

    for outputs in all_outputs:
        for output in outputs.outputs:
            text = (
                tokenizer.decode(output.token_ids, skip_special_tokens=True)
                .replace(" ", "")
                .replace("\n", "")
            )
            generated_sequences.append(text)
            lp = np.mean(
                [max([p.logprob for p in prob.values()]) for prob in output.logprobs]
            )
            scores.append(lp)

    print(f"Generated {len(generated_sequences)} sequences")

    return generated_sequences, scores


def main(args: dict):
    # Initialize the SequenceGenerator with the provided arguments
    random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    kws = {
        "model_dir": args["model_dir"],
        "promptfile": args["prompt"],
        "min_seq_len": args["min_seq_len"],
        "max_seq_len": args["seq_len"],
        "temperature": args["temperature"],
        "top_k": args["top_k"],
        "top_p": args["top_p"],
        "presence_penalty": args["presence_penalty"],
        "frequency_penalty": args["frequency_penalty"],
        "repetition_penalty": args["repetition_penalty"],
        "seed": args["seed"],
    }
    seq_gen = SequenceGenerator(num=args["num"], **kws)
    if not args["batch_size"]:
        output_seqs, output_scores = generate(seq_gen)
    else:
        output_seqs, output_scores = [], []
        for batch in create_batches(args["num"], args["batch_size"]):
            cur_seqs, cur_scores = generate(seq_gen, batch)
            output_seqs.extend(cur_seqs)
            output_scores.extend(cur_scores)

    if Path(args["prompt"]).exists():
        prompt = str(SeqIO.read(args["prompt"], "fasta").seq)
    else:
        prompt = args["prompt"]
    if args["prepend_prompt_to_output"]:
        print("Prepending prompt...")
        output_seqs = [prompt + s for s in output_seqs if not s.startswith(prompt)]

    with open(args["output"], "w") as f:
        to_write = [f">{args['prefix']}{i}\n{seq}" for i, seq in enumerate(output_seqs)]
        f.write("\n".join(to_write))
    if args["output_scores"]:
        with open(args["output_scores"], "w") as f:
            writer = DictWriter(f, ["id", "score"])
            writer.writeheader()
            for i, score in enumerate(output_scores):
                writer.writerow({"id": f"{args['prefix']}{i}", "score": score})


if __name__ == "__main__":
    args = parse_args()
    main(args)
