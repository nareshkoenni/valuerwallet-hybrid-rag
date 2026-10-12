#!/usr/bin/env python3
"""Recompute retrieval metrics from a saved ranking file.

Works on the ranking files written during the study (``*_top20.csv``): one row per
retrieved candidate, or one row per query, with the columns ``query_index``,
``question_id`` and ``target_rank`` (empty when the target is not in the top 20).
The ranking files contain real file numbers and are not distributed in this
repository (see DATA_AVAILABILITY.md); authorised reviewers can use this script to
check the metric tables in ``results/`` against them.

    python scripts/recompute_metrics.py PATH/TO/semantic_baseline_v2_top20.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vwrag.metrics import summarise  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rankings")
    parser.add_argument("--out", help="optional CSV path for the metric table")
    args = parser.parse_args()

    cols = ["query_index", "question_id", "target_rank"]
    frame = pd.read_csv(args.rankings, usecols=cols, dtype={"question_id": str})
    per_query = frame.drop_duplicates("query_index").rename(columns={"target_rank": "rank_x"})
    table = summarise(per_query, "rank_x")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
