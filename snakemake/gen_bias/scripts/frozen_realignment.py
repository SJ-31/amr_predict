#!/usr/bin/env python3

import argparse
from collections.abc import Iterable
from multiprocessing import Pool

import pytest
from Bio import Align, SeqIO
from Bio.Align import Alignment
from Bio.Seq import Seq
from Bio.SeqIO import SeqRecord
from pyhere import here

test_dir = here("data", "remote", "tests", "test_realignment")
test_msa = test_dir / "aligned.fasta"
test_query = test_dir / "to_align.fasta"


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


def realign_one(
    query: SeqRecord, msa: list[SeqRecord], threshold: float = 0.99
) -> SeqRecord:
    aligner = semi_global_aligner()
    scores: list[tuple[tuple[str, str], int]] = []
    best: tuple[str, str] | None = None
    best_index: int = -1
    for i, seq in enumerate(msa):
        degapped = seq.seq.replace("-", "")
        target_a, query_a, score = align_and_pad(
            target=degapped, query=query, aligner=aligner
        )
        if not score:
            continue
        if (score / len(query)) >= threshold:
            best, best_index = (target_a, query_a), i
            break
        scores.append(((target_a, query_a), i))
    if not best:
        best, best_index = max(scores, key=lambda x: x[0].score)
    degapped, query = best
    msa_str = str(msa[best_index].seq)
    realigned = insert_msa_gaps(query=query, degapped=degapped, msa_seq=msa_str)
    return SeqRecord(Seq(realigned), id=query.id)


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
        "--batch_size",
        default=None,
        help="Custom batch size for each worker process when threads > 1",
        action="store",
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
        default=1,
        help="> 1 processes query sequences in parallel",
        action="store",
        type=int,
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


def main(args):
    msa = [s.upper() for s in SeqIO.parse(args["msa"], format="fasta")]
    queries = validate_queries(SeqIO.parse(args["query"], "fasta"))
    if args["threads"] > 1:
        with Pool(args["threads"]) as pool:
            result = pool.imap(
                lambda q: realign_one(q, msa, args["threshold"]),
                queries,
                chunksize=args["batch_size"],
            )
    else:
        result = [realign_one(q, msa, args["threshold"]) for q in queries]


if __name__ == "__main__":
    args = parse_args()


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
