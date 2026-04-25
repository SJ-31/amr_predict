#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# sys.path.append("/py_lib")
import click
import torch
from loguru import logger

logger.info("PYTHONPATH {}", os.environ.get("PYTHONPATH", "Nothing"))
logger.info("sys.path {}", sys.path)
logger.info("cuda available {}", torch.cuda.is_available())

from amr_predict.cache import EmbeddingCache, LinkedDataset


@click.command()
@click.option("-o", "--output", required=True, help="snakmake output", default=None)
def shell_rule(output):
    Path(output).write_text("hello world")
