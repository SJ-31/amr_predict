#!/usr/bin/env ipython

from collections.abc import Sequence

import polars as pl
import torch


def torch2pl(
    dtype: torch.dtype | Sequence[torch.dtype],
) -> pl.DataType | list[pl.DataType]:
    # Valid v2.9.1
    mapping: dict = {
        torch.bool: pl.Boolean,
        torch.int8: pl.Int8,
        torch.uint8: pl.UInt8,
        torch.int16: pl.Int16,  # alias: torch.short
        torch.uint16: pl.UInt16,
        torch.int32: pl.Int32,  # alias: torch.int
        torch.uint32: pl.UInt32,
        torch.int64: pl.Int64,  # alias: torch.long
        torch.uint64: pl.UInt64,
        torch.float32: pl.Float32,  # alias: torch.float
        torch.float64: pl.Float64,  # alias: torch.double
    }
    if isinstance(dtype, Sequence):
        converted = []
        for tp in dtype:
            try:
                converted.append(mapping[tp])
            except KeyError:
                raise ValueError(f"`{tp}` is not supported by polars")
        return converted
    try:
        return mapping[dtype]
    except KeyError:
        raise ValueError(f"`{dtype}` is not supported by polars")


def torch2hf(dtype: torch.dtype | Sequence[torch.dtype]) -> Value | list[Value]:
    # Valid v2.9.1
    mapping: dict = {
        torch.bool: "bool",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.int16: "int16",  # alias: torch.short
        torch.uint16: "uint16",
        torch.int32: "int32",  # alias: torch.int
        torch.uint32: "uint32",
        torch.int64: "int64",  # alias: torch.long
        torch.uint64: "uint64",
        torch.float16: "float16",  # alias: torch.half
        torch.float32: "float32",  # alias: torch.float
        torch.float64: "float64",  # alias: torch.double
    }
    if isinstance(dtype, Sequence):
        converted = []
        for tp in dtype:
            try:
                converted.append(Value(mapping[tp]))
            except KeyError:
                raise ValueError(f"`{tp}` is not supported by HF")
        return converted
    try:
        return Value(mapping[dtype])
    except KeyError:
        raise ValueError(f"`{dtype}` is not supported by HF")
