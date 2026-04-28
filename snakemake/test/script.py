#!/usr/bin/env python3

import os
import sys

sys.path.append("/py_lib")
from pathlib import Path

import torch

text = f"""
PYTHONPATH {os.environ.get("PYTHONPATH", "Nothing")}
sys.path {sys.path}
cuda available {torch.cuda.is_available()}
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake.iocontainers import snakemake

text = text + f"\nsmk config {snakemake.config}"

from amr_predict.cache import EmbeddingCache, LinkedDataset

Path(snakemake.output[0]).write_text(text)
