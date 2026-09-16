#!/usr/bin/env python3


import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizer

repo_id = "InstaDeepAI/NTv3_8M_pre"

tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(repo_id, trust_remote_code=True)
model.eval()


def mdlm_sample(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    t: float,
    dt: float,
    mask_token_id: int,
    valid_token_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Sample from MDLM posterior distribution.

    Args:
        tokens: (B, L) current tokens
        logits: (B, L, V) model logits
        t: current time step (1 -> 0)
        dt: time step size
        mask_token_id: ID of mask token
        valid_token_ids: list of valid token IDs to sample from (e.g., A, C, G, T)

    Returns:
        (B, L) sampled tokens
    """
    # Linear noise schedule: noise_level = t
    noise_t = t
    noise_s = max(t - dt, 0)
    signal_t = 1.0 - noise_t
    signal_s = 1.0 - noise_s

    # Mask invalid tokens if specified
    if valid_token_ids is not None:
        mask = torch.full_like(logits, float("-inf"))
        mask[..., valid_token_ids] = logits[..., valid_token_ids]
        logits = mask

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)

    # Compute posterior for masked positions
    # masked_posterior = probs * (signal_s - signal_t) / (1 - signal_t) + mask_prob
    masked_posterior = probs * (signal_s - signal_t) / (1 - signal_t + 1e-8)
    masked_posterior[..., mask_token_id] = (1 - signal_s) / (1 - signal_t + 1e-8)

    # Copy-over for unmasked positions (keep original token)
    unmasked_posterior = F.one_hot(tokens, logits.shape[-1]).float()

    # Combine: use masked_posterior where token is mask, else unmasked_posterior
    posterior = torch.where(
        tokens[..., None] == mask_token_id, masked_posterior, unmasked_posterior
    )

    # Sample from posterior
    dist = Categorical(probs=posterior)
    return dist.sample()


# =============================================================================
# Multi-Context CFG Generation (Promoter-Specific with Classifier-Free Guidance)
# =============================================================================


@torch.no_grad()
def generate_enhancer_cfg(
    model: nn.Module,
    exp_tokens: torch.Tensor,
    exp_species_ids: torch.Tensor,
    exp_activity_ids: torch.Tensor,
    exp_insert_pos: Tuple[int, int],
    bg_tokens: torch.Tensor,
    bg_species_ids: torch.Tensor,
    bg_activity_ids: torch.Tensor,
    bg_insert_pos: Tuple[int, int],
    mask_token_id: int,
    valid_token_ids: List[int],
    gamma: float = 12,
    num_steps: int = 100,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate enhancer using Classifier-Free Guidance with two backbone contexts.

    This generates an enhancer that works well in the "experiment" context while
    contrasting against the "background" context. Higher gamma pushes more strongly
    toward the experiment condition.

    Args:
        model: trained generation model
        exp_tokens: (B, L) experiment backbone with masked enhancer region
        exp_species_ids: (B,) species tokens for experiment context
        exp_activity_ids: (B,) activity tokens for experiment context
        exp_insert_pos: (start, end) tuple for enhancer position in exp backbone
        bg_tokens: (B, L) background backbone with masked enhancer region
        bg_species_ids: (B,) species tokens for background context
        bg_activity_ids: (B,) activity tokens for background context
        bg_insert_pos: (start, end) tuple for enhancer position in bg backbone
        mask_token_id: ID of mask token
        valid_token_ids: list of valid nucleotide token IDs
        gamma: CFG strength (>1 pushes toward experiment, 1=no guidance)
        num_steps: number of diffusion steps
        device: 'cuda' or 'cpu'

    Returns:
        (exp_tokens, bg_tokens) - both backbones with generated enhancer
    """
    model.eval()
    model = model.to(device)
    exp_activity_cond_mask = torch.where(exp_activity_ids != -1, 1, 0).to(torch.bool)
    bg_activity_cond_mask = torch.where(bg_activity_ids != -1, 1, 0).to(torch.bool)
    exp_activity_ids = torch.where(exp_activity_ids == -1, 0, exp_activity_ids)
    bg_activity_ids = torch.where(bg_activity_ids == -1, 0, bg_activity_ids)

    exp_tokens = exp_tokens.to(device)
    bg_tokens = bg_tokens.to(device)
    exp_species_ids = exp_species_ids.to(device)
    bg_species_ids = bg_species_ids.to(device)
    exp_activity_ids = exp_activity_ids.to(device)
    exp_activity_cond_mask = exp_activity_cond_mask.to(device)
    bg_activity_ids = bg_activity_ids.to(device)
    bg_activity_cond_mask = bg_activity_cond_mask.to(device)

    dt = 1.0 / num_steps
    time_steps = torch.linspace(1, 0, num_steps + 1)

    # Conditioning masks (always use conditioning)
    B = exp_tokens.shape[0]
    species_cond_mask = torch.ones(B, dtype=torch.bool, device=device)

    exp_start, exp_end = exp_insert_pos
    bg_start, bg_end = bg_insert_pos

    for t in time_steps[:-1]:
        t_val = t.item()

        # Forward pass for experiment context
        exp_outputs = model(
            input_ids=exp_tokens,
            condition_ids=[exp_species_ids, exp_activity_ids],
            conditions_masks=[species_cond_mask, exp_activity_cond_mask],
        )
        exp_logits = exp_outputs["logits"][:, exp_start:exp_end]

        # Forward pass for background context
        bg_outputs = model(
            input_ids=bg_tokens,
            condition_ids=[bg_species_ids, bg_activity_ids],
            conditions_masks=[species_cond_mask, bg_activity_cond_mask],
        )
        bg_logits = bg_outputs["logits"][:, bg_start:bg_end]

        # CFG: combine logits
        # combined = (1 - gamma) * background + gamma * experiment
        combined_logits = (1.0 - gamma) * bg_logits + gamma * exp_logits

        # Sample from combined posterior
        enhancer_tokens = exp_tokens[:, exp_start:exp_end]
        new_enhancer = mdlm_sample(
            enhancer_tokens, combined_logits, t_val, dt, mask_token_id, valid_token_ids
        )

        # Update both backbones with the same enhancer
        exp_tokens = exp_tokens.clone()
        bg_tokens = bg_tokens.clone()
        exp_tokens[:, exp_start:exp_end] = new_enhancer
        bg_tokens[:, bg_start:bg_end] = new_enhancer

    return exp_tokens, bg_tokens


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prefix: str,
    device: str = "cpu",
    num_steps: int = 100,
    length: int = 100,
    gamma: float = 12,
    mask: str = "<mask>",
) -> torch.Tensor:
    model.eval()
    model.to(device)
    tokenized = tokenizer(
        prefix + mask * length,
        add_special_tokens=False,
        padding=True,
        pad_to_multiple_of=128,
        return_tensors="pt",
    )
    time_steps = torch.linspace(1, 0, num_steps + 1)
    # Can do this in generic way by appending a single mask token at each
    # round
    input_ids: torch.Tensor = tokenized["input_ids"]
    fill_start, fill_end = len(prefix), len(prefix) + length
    input_ids[:, fill_start:fill_end] = tokenizer.mask_token_id
    valid_token_ids = [tokenizer.convert_tokens_to_ids(b) for b in ["A", "C", "G", "T"]]
    dt = 1 / num_steps
    for t in time_steps[:-1]:
        t_val = t.item()
        out = model(input_ids=input_ids)
        logits = out["logits"][:, fill_start:fill_end]
        logits = gamma * logits
        sampled = mdlm_sample(
            input_ids[:, fill_start:fill_end],
            logits=logits,
            t=t_val,
            dt=dt,
            mask_token_id=tokenizer.mask_token_id,
            valid_token_ids=valid_token_ids,
        )
        input_ids[:, fill_start:fill_end] = sampled
    return input_ids


# TODO: [2026-09-16 Wed] refactor so that it can generate multiple sequences

# TODO: [2026-09-16 Wed] this works, but what is the equivalent of
# temperature in MDLM? And what other sampling parameters to vary?


def generate_enhancers_batch(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    species_dict: Dict[str, int],
    backbone_metadata: Dict[str, Any],
    # Experiment context
    exp_promoter: str,
    exp_activity: int,
    # Background context
    bg_promoter: str,
    bg_activity: int,
    # Generation parameters
    num_sequences: int = 10,
    batch_size: int = 4,
    num_steps: int = 50,
    gamma: float = 2.0,
    device: Optional[str] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Generate enhancer sequences using multi-context CFG diffusion.

    Args:
        model: Trained generation model
        tokenizer: Sequence tokenizer
        species_dict: Species dictionary
        backbone_metadata: Dict with promoter backbone info
        exp_promoter: Experiment promoter name ('DSCP' or 'RpS12')
        exp_activity: Experiment activity level (0-4)
        bg_promoter: Background promoter name ('DSCP' or 'RpS12')
        bg_activity: Background activity level (0-4)
        num_sequences: Total number of sequences to generate
        batch_size: Batch size for parallel generation
        num_steps: Number of diffusion steps
        gamma: CFG guidance strength (>1 pushes toward experiment)
        device: 'cuda' or 'cpu' (auto-detect if None)
        verbose: Print progress messages

    Returns:
        List of generated enhancer sequences (strings)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get promoter info
    exp_info = backbone_metadata["promoters"][exp_promoter]
    bg_info = backbone_metadata["promoters"][bg_promoter]
    exp_insert = (exp_info["enhancer_insert_start"], exp_info["enhancer_insert_end"])
    bg_insert = (bg_info["enhancer_insert_start"], bg_info["enhancer_insert_end"])

    if verbose:
        print(f"Experiment: {exp_promoter} + Activity {exp_activity}")
        print(f"Background: {bg_promoter} + Activity {bg_activity}")
        print(
            f"Generating {num_sequences} sequences (batch_size={batch_size}, gamma={gamma})"
        )

    # Token IDs
    mask_token_id = tokenizer.mask_token_id
    valid_token_ids = [tokenizer.convert_tokens_to_ids(b) for b in ["A", "C", "G", "T"]]

    # Tokenize backbone templates
    exp_template = tokenizer(
        exp_info["sequence"], add_special_tokens=False, return_tensors="pt"
    )["input_ids"]
    exp_template[0, exp_insert[0] : exp_insert[1]] = mask_token_id

    bg_template = tokenizer(
        bg_info["sequence"], add_special_tokens=False, return_tensors="pt"
    )["input_ids"]
    bg_template[0, bg_insert[0] : bg_insert[1]] = mask_token_id

    # Species ID
    species_id = torch.tensor(species_dict["drosophila_melanogaster"])

    # Generate in batches
    all_enhancers = []
    num_batches = (num_sequences + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_sequences - len(all_enhancers))

        # Expand to batch
        exp_tokens = exp_template.repeat(current_batch_size, 1)
        bg_tokens = bg_template.repeat(current_batch_size, 1)
        exp_species = species_id.expand(current_batch_size)
        bg_species = species_id.expand(current_batch_size)
        exp_acts = torch.full((current_batch_size,), exp_activity, dtype=torch.long)
        bg_acts = torch.full((current_batch_size,), bg_activity, dtype=torch.long)

        # Generate
        exp_result, _ = generate_enhancer_cfg(
            model=model,
            exp_tokens=exp_tokens,
            exp_species_ids=exp_species,
            exp_activity_ids=exp_acts,
            exp_insert_pos=exp_insert,
            bg_tokens=bg_tokens,
            bg_species_ids=bg_species,
            bg_activity_ids=bg_acts,
            bg_insert_pos=bg_insert,
            mask_token_id=mask_token_id,
            valid_token_ids=valid_token_ids,
            gamma=gamma,
            num_steps=num_steps,
            device=device,
        )

        # Decode enhancers
        for i in range(current_batch_size):
            enhancer_tokens = exp_result[i, exp_insert[0] : exp_insert[1]]
            enhancer_seq = tokenizer.decode(enhancer_tokens, skip_special_tokens=True)
            all_enhancers.append(enhancer_seq)

        if verbose:
            print(
                f"  Batch {batch_idx + 1}/{num_batches}: {len(all_enhancers)}/{num_sequences}"
            )

    if verbose:
        print(f"Done! Generated {len(all_enhancers)} enhancers.")

    return all_enhancers
