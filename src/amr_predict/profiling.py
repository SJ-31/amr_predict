#!/usr/bin/env ipython
from collections.abc import Callable
from pathlib import Path
from subprocess import run

import memray


def memray_from_smk(env: dict, fn, outfile, env_key: str = "memray"):
    """Call top-level function `fn` with memray profiling if it is set up in the
    snakemake configuration

    Parameters
    ----------
    env_key : str
        Top-level key in snakemake configuration describing how to run memray
        Should have the following structure

    ENV_KEY:
      run: False
      flamegraph:
          run: True
          kws: # Any valid flags for the `memray flamegraph` CLI command
            temporal: True
            ...
      table:
          run: True
          kws:
            leaks: True
            ...
    """
    cfg = env[env_key]
    if cfg.get("run"):
        print(f"Running `{fn.__name__}` with memray profiling")
        fg: dict = cfg.get("flamegraph", {})
        tb: dict = cfg.get("table", {})
        memray_wrapper(
            fn,
            outfile=outfile,
            flamegraph=fg.get("run"),
            flamegraph_kws=fg.get("kws", {}),
            table=tb.get("run"),
            table_kws=tb.get("kws", {}),
        )
    else:
        fn()


def memray_wrapper(
    fn: Callable,
    bin_out: str,
    flamegraph: bool = True,
    table: bool = True,
    flamegraph_kws: dict | None = None,
    table_kws: dict | None = None,
    **kws,
):
    try:
        with memray.Tracker(bin_out, **kws):
            fn()
    finally:
        bfile = Path(bin_out)
        for suffix, command, should_run, kws in zip(
            ("flamegraph", "table"),
            (memray_flamegraph, memray_table),
            (flamegraph_kws, table_kws),
            (flamegraph, table),
        ):
            if should_run:
                out = bfile.parent / f"{bfile.stem}-{suffix}.html"
                proc = run(command(bfile, out, **kws), shell=True)
                proc.check_returncode()


def memray_table(
    input_file: Path | str,
    out: Path | str,
    force: bool = False,
    leaks: bool = False,
    temporary_allocation_threshold: int = -1,
    temporary_allocations: bool = False,
    no_web: bool = False,
) -> list:
    cmd = ["memray", "table", str(input_file)]
    cmd.extend(
        [
            "--output",
            str(out),
            "--temporary-allocation-threshold",
            temporary_allocation_threshold,
        ]
    )
    if temporary_allocations:
        cmd.append("--temporary-allocations")
    if leaks:
        cmd.append("--leaks")
    if force:
        cmd.append("--force")
    if no_web:
        cmd.append("--no_web")
    return cmd


def memray_flamegraph(
    input_file: Path | str,
    out: Path | str,
    force: bool = False,
    temporal: bool = False,
    split_threads: bool = False,
    leaks: bool = False,
    inverted: bool = False,
    temporary_allocation_threshold: int = -1,
    temporary_allocations: bool = False,
    no_web: bool = False,
) -> list:
    cmd = ["memray", "flamegraph", str(input_file)]
    cmd.extend(
        [
            "--output",
            str(out),
            "--temporary-allocation-threshold",
            temporary_allocation_threshold,
        ]
    )
    if temporal:
        cmd.append("--temporal")
    if temporary_allocations:
        cmd.append("--temporary-allocations")
    if split_threads:
        cmd.append("--split-threads")
    if leaks:
        cmd.append("--leaks")
    if force:
        cmd.append("--force")
    if inverted:
        cmd.append("--inverted")
    if no_web:
        cmd.append("--no_web")
    return cmd
