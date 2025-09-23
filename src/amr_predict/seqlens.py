#!/usr/bin/env python

# Utilities for working with seqlens models

import os

import torch

# tokenizer = AutoTokenizer.from_pretrained("omicseye/seqLens_4096_512_46M-Mp")
# model = AutoModelForMaskedLM.from_pretrained("omicseye/seqLens_4096_512_46M-Mp")
from datasets import load_dataset

os.environ["HF_HOME"] = ""


def extract_representations(model, dataloader, device="cpu", count=5):
    """
    Extract mean, CLS, and max representations from a model's final hidden layer.

    Args:
        model (torch.nn.Module): The pre-trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for input data.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        dict: A dictionary containing mean, CLS, and max representations as Tensors.
    """
    # Move model to device and set to eval mode
    model.to(device)
    model.eval()

    # Initialize lists to store outputs
    mean_outputs = []
    cls_outputs = []
    max_outputs = []

    # Process batches
    out = []
    with torch.no_grad():
        for batch in dataloader:
            # Move inputs to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get model outputs
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            out.append(outputs)
            if len(out) == count:
                return out

    #         last_hidden_state = outputs.last_hidden_state

    #         # Mask out padding tokens
    #         expanded_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state)
    #         hidden_state_masked = last_hidden_state * expanded_mask

    #         # Compute CLS token representation
    #         cls_token_representation = last_hidden_state[:, 0, :]  # CLS token
    #         cls_outputs.append(cls_token_representation.cpu())

    #         # Compute mean pooling (excluding padding tokens)
    #         sum_hidden_states = hidden_state_masked.sum(dim=1)
    #         non_padding_tokens = attention_mask.sum(dim=1).unsqueeze(-1)
    #         mean_representation = sum_hidden_states / non_padding_tokens
    #         mean_outputs.append(mean_representation.cpu())

    #         # Compute max pooling
    #         max_representation, _ = hidden_state_masked.max(dim=1)
    #         max_outputs.append(max_representation.cpu())
    return out

    # # Concatenate all outputs
    # mean_outputs = torch.cat(mean_outputs, dim=0)
    # cls_outputs = torch.cat(cls_outputs, dim=0)
    # max_outputs = torch.cat(max_outputs, dim=0)

    # print("Processing complete!")
    # print(f"Mean outputs shape: {mean_outputs.shape}")
    # print(f"CLS outputs shape: {cls_outputs.shape}")
    # print(f"Max outputs shape: {max_outputs.shape}")

    # return {"mean": mean_outputs, "cls": cls_outputs, "max": max_outputs}


tokenized_dataset = test_dataset.map(
    tokenize_function, batched=True, remove_columns=["Sequence", "Class", "db"]
)
# Need to remove any string columns in this operation
