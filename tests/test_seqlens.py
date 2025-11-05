#!/usr/bin/env ipython

import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding

os.environ["HF_HOME"] = (
    "/data/project/stemcell/shannc/repos/amr_predict/cache/huggingface"
)

model_key: str = "omicseye/seqLens_4096_512_46M-Mp"
tokenizer = AutoTokenizer.from_pretrained(model_key)
model = AutoModelForMaskedLM.from_pretrained(model_key)
model.config.output_hidden_states = True
ds = load_dataset("omicseye/novel_arg_eval", split="train")


def tokenize_fn(data):
    return tokenizer(
        data["Sequence"],
        truncation=True,
        padding=False,
        max_length=512,
    )


tokenized = ds.map(
    tokenize_fn, batched=True, remove_columns=["Sequence", "Class", "db"]
)
# Removing text cols is required

outputs = []
count = 5
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
loader = DataLoader(tokenized, batch_size=32, collate_fn=data_collator)

acc = 0
with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"]
        print(batch["input_ids"].shape)
        attention_mask = batch["attention_mask"]
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        outputs.append(output)
        if acc == count:
            break
        acc += 1

mask = torch.tensor(tokenized[0]["attention_mask"]).reshape(1, -1)
sample = model(
    input_ids=torch.tensor(tokenized[0]["input_ids"]).reshape(1, -1),
    attention_mask=mask,
)
hidden = sample.hidden_states[-1]
expanded_mask = mask.unsqueeze(-1).expand_as(hidden)
hidden_masked = hidden * expanded_mask
