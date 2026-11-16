#!/usr/bin/env python3

import argparse
from collections.abc import Iterable
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from subprocess import PIPE, run
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Literal

import polars as pl
import pytest
from Bio import Align, SeqIO
from Bio.Align import Alignment
from Bio.Seq import Seq
from Bio.SeqIO import SeqRecord

"""
Insert new sequences into an MSA without changing the existing alignment i.e. gaps are only added to the new query sequences.
The resulting MSA is guaranteed to have the same number of columns as the input


Steps
1. Given a query sequence and msa, loop over lines of msa and remove padding to get natural sequence.
2. Perform semi-global alignment with the msa sequence and the query. Repeat for every seq in msa and identify the top-scoring sequence by seq identity.
3. Retain the top alignment, insert gaps in the query according to how the sequence was gapped in the MSA.
4. Repeat for every query sequence
"""

PARASAIL_COLS: list[str] = [
    "query",
    "index",
    "length1",
    "length2",
    "score",
    "end_query",
    "end_ref",
    "n_matches",
    "similarities",
    "length",
]


def insert_msa_gaps(query: str, degapped: str, msa_seq: str) -> str:
    tmp = ["-"] * len(msa_seq)
    msa_index: int = 0
    for i in range(len(degapped)):
        while msa_seq[msa_index] == "-":
            tmp[msa_index] = "-"
            msa_index += 1
        tmp[msa_index] = query[i]
        msa_index += 1
    return "".join(tmp)


def align_and_pad(
    target, query, aligner: Align.PairwiseAligner
) -> tuple[str, str, float]:
    """
    Wrapper around PairwiseAligner.align that pads the query sequence with gaps
    so that the aligned results are
    guaranteed to have the same length as the target sequence

    Returns
    -------
    tuple of [aligned_target, aligned_query, alignment_score]
    """
    obj = aligner.align(target, query)
    if not obj.score:
        return "", "", 0
    alignment: Alignment = next(obj)
    target_a, query_a = alignment[:2]
    if len(target_a) != len(target):
        target_coords, query_coords = alignment.coordinates
        target_a = (
            target[: min(target_coords)]
            + target_a
            + target[max(target_coords) : len(target) + 1]
        )
        query_a = (
            ("-" * min(target_coords))
            + query_a
            + ("-" * (len(target) - max(target_coords)))
        )
    return target_a, query_a, alignment.score


def semi_global_aligner() -> Align.PairwiseAligner:
    """
    Returns
    -------
    PairwiseAligner object parameterized for semi-global alignment of query to target

    Notes
    -----
    - `insertion` refers to creating gaps in `target`, `deletion` for gaps in `query`
       See https://biopython.org/docs/dev/Tutorial/chapter_pairwise.html#affine-gap-scores
    - the call syntax is `align(target, query)`
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 1
    aligner.open_insertion_score = float("-inf")  # gap open
    aligner.extend_insertion_score = float("-inf")  # gap extension
    aligner.open_right_deletion_score = 0
    aligner.extend_right_deletion_score = 0
    aligner.open_left_deletion_score = 0
    aligner.extend_left_deletion_score = 0
    return aligner


def realign_one_parasail(
    query: SeqRecord,
    degapped_msa: Path,
    msa: list[SeqRecord],
    threads: int = 1,
    batch_size: int = 50,
    data_type: Literal["dna", "aa"] = "dna",
) -> SeqRecord:
    type_flag = "-d" if data_type == "dna" else ""
    with TemporaryDirectory() as d:
        tmp: Path = Path(d)
        SeqIO.write(query, tmp / "query.fasta", format="fasta")
        args = [
            "parasail_aligner",
            "-q",
            tmp / "query.fasta",
            "-f",
            degapped_msa,
            "-a",
            "sw_stats_striped_16",
            "-t",
            threads,
            "-b",
            batch_size,
            type_flag,
            "-g",
            tmp / "output.csv",
        ]
        proc = run(" ".join([str(a) for a in args]), shell=True, stderr=PIPE)
        proc.check_returncode()
        results: pl.DataFrame = pl.read_csv(
            tmp / "output.csv", has_header=False, new_columns=PARASAIL_COLS
        )
    best_idx: int = results.top_k(1, by="score")["index"][0]
    best_seq: SeqRecord = msa[best_idx]
    t_a, q_a, _ = align_and_pad(
        target=best_seq.seq.replace("-", ""), query=query, aligner=semi_global_aligner()
    )
    realigned = insert_msa_gaps(query=q_a, degapped=t_a, msa_seq=str(best_seq.seq))
    result = SeqRecord(Seq(realigned), id=query.id, description=query.description or "")
    assert len(result) == len(best_seq)
    return result


def realign_one(
    query: SeqRecord, msa: list[SeqRecord], threshold: float = 0.99
) -> SeqRecord:
    aligner = semi_global_aligner()
    scores: list[tuple[tuple[str, str], int, float]] = []
    best: tuple[str, str] | None = None
    best_index: int = -1
    target_length = len(msa[0])
    for i, seq in enumerate(msa):
        degapped = seq.seq.replace("-", "")
        t_a, q_a, score = align_and_pad(target=degapped, query=query, aligner=aligner)
        if not score:
            continue
        if (score / len(query)) >= threshold:
            best, best_index = (t_a, q_a), i
            break
        scores.append(((t_a, q_a), i, score))
    if not best:
        best, best_index, _ = max(scores, key=lambda x: x[2])
    target_a, query_a = best
    msa_str = str(msa[best_index].seq)
    realigned = insert_msa_gaps(query=query_a, degapped=target_a, msa_seq=msa_str)
    res = SeqRecord(Seq(realigned), id=query.id, description=query.description or "")
    assert len(res) == target_length
    return res


def validate_queries(queries: Iterable[SeqRecord], max_len: int) -> list[SeqRecord]:
    tmp = []
    for i, q in enumerate(queries):
        if len(q) > max_len:
            msg = f"Problem sequence: 0-based index {i}"
            if id := q.metadata.get("id"):
                msg = f"{msg}, id: {id}"
            raise ValueError(
                f"No sequence in `queries` can be longer than the MSA.\n{msg}"
            )
        tmp.append(q.upper())
    return tmp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q", "--query", help="Query sequences in FASTA format", required=True
    )
    parser.add_argument("-m", "--msa", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument(
        "-b",
        "--batch-size",
        default=[50, 50],
        help="""
        1st specification: Custom batch size for each worker process when threads > 1
        2nd specification: batch size for parasail_aligner (if --parasail)
        """,
        action="append",
        type=int,
    )
    parser.add_argument(
        "-d",
        "--threshold",
        default=0.99,
        help="Identity threshold at which to stop searching for better alignments",
        action="store",
        type=float,
    )
    parser.add_argument(
        "-t",
        "--threads",
        default=[1, 1],
        help="""
        1st specification: setting to > 1 processes query sequences in parallel with the
            multiprocessing library
        2nd specification: number of threads passed to parasail_aligner (if --parasail)
        """,
        action="append",
        type=int,
    )
    parser.add_argument(
        "-p",
        "--parasail",
        default=False,
        help="Whether to use parasail_aligner for pairwise alignments",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--seqtype",
        choices=["dna", "rna", "protein"],
        help="Type of sequence",
        default="dna",
    )
    args = vars(parser.parse_args())  # convert to dict
    return args


def main(args, msa: list[SeqRecord]) -> list[SeqRecord]:
    queries = validate_queries(SeqIO.parse(args["query"], "fasta"), max_len=len(msa[0]))
    if len(args["threads"]) < 2:
        args["threads"].append(1)
    if len(args["batch_size"]) < 2:
        args["batch_size"].append(50)
    threads, threads_p = args["threads"][:2]
    batch_size, batch_size_p = args["threads"][:2]
    if args["parasail"]:
        with NamedTemporaryFile("w") as f:
            degapped = "\n".join(
                [f">{seq.description}\n{seq.seq.replace('-', '')}" for seq in msa]
            )
            f.write(degapped)
            f.flush()
            fn = partial(
                realign_one_parasail,
                degapped_msa=f.name,
                msa=msa,
                threads=threads_p,
                batch_size=batch_size_p,
                data_type=args["seqtype"],
            )
            if threads > 1:
                with Pool(threads) as pool:
                    mapped = pool.imap(fn, queries, chunksize=batch_size)
                    result = list(mapped)
            else:
                result = [fn(q) for q in queries]
    elif threads > 1:
        with Pool(threads) as pool:
            mapped = pool.imap(
                lambda q: realign_one(q, msa, args["threshold"]),
                queries,
                chunksize=batch_size,
            )
            result = list(mapped)
    else:
        result = [realign_one(q, msa, args["threshold"]) for q in queries]
    return result


if __name__ == "__main__":
    args = parse_args()
    msa = [s.upper() for s in SeqIO.parse(args["msa"], format="fasta")]
    realigned = main(args, msa)
    combined = msa + realigned
    SeqIO.write(combined, args["output"], "fasta")


@pytest.mark.parametrize(
    "query,degapped,msa_seq,ans",
    [
        (
            "--DTGZFDCA--TG",
            "FSDTGAFDCADFTA",
            "F--SDTGAF--DCAD--FTA",
            "----DTGZF--DCA----TG",
        ),
        (
            "-HOWTAT--E-W-ERE",
            "WHOWHATWHENWHERE",
            "---WHO--WHAT-WHEN--WHERE---",
            "----HO--WTAT---E---W-ERE---",
        ),
    ],
)
def test_insert_gaps(query, degapped, msa_seq, ans):
    result = insert_msa_gaps(query, degapped, msa_seq)
    assert len(result) == len(msa_seq)
    print(f"RES: {result}\nANS: {ans}")
    assert result == ans


@pytest.mark.parametrize(
    "q,t,padded",
    [
        ("whatwhen", "whowhatwhenwhere", "---whatwhen-----"),
        ("bazbo", "barbazbatbuzz", "---bazb------"),
        ("789", "123456789", "------789"),
        ("123", "123456789", "123------"),
    ],
)
def test_align_pad(q, t, padded):
    aligner = semi_global_aligner()
    t_a, q_a, _ = align_and_pad(t, q, aligner)
    assert t_a == t
    assert q_a == padded


@pytest.mark.parametrize(
    "threads,parasail", [(2, False), (1, False), (2, True), (1, True)]
)
def test_main(threads, parasail):
    from pyhere import here

    test_dir = here("data", "remote", "tests", "test_realignment")
    test_msa = test_dir / "mini_aligned.fasta"
    test_query = test_dir / "to_align.fasta"
    msa = [s.upper() for s in SeqIO.parse(test_msa, format="fasta")]
    args = {
        "query": test_query,
        "threads": [threads],
        "threshold": 0.99,
        "batch_size": [5],
        "parasail": parasail,
        "seqtype": "dna",
    }
    result = main(args, msa)
    print(result)
