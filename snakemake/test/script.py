#!/usr/bin/env python3

import os
import sys

# sys.path.append("/py_lib")
from pathlib import Path

import torch
from loguru import logger

logger.info("PYTHONPATH {}", os.environ.get("PYTHONPATH", "Nothing"))
logger.info("sys.path {}", sys.path)
logger.info("cuda available {}", torch.cuda.is_available())


try:
    from snakemake.script import snakemake as smk
except ImportError:
    smk = type("snakemake", (), {"rule": None, "config": {}, "log": [0]})
logger.info("smk config {}", smk.config)

from amr_predict.cache import EmbeddingCache, LinkedDataset

Path(smk.output[0]).write_text("hello world")
