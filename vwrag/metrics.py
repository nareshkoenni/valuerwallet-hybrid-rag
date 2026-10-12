"""Rank-based retrieval metrics for single-target queries."""

from __future__ import annotations

import numpy as np
import pandas as pd

K_VALUES = (5, 10, 20)


def metrics_from_ranks(ranks: pd.Series, k_max: int = 20) -> dict[str, float]:
    """Recall@K and MRR@k_max from 1-based target ranks (NaN = target not retrieved)."""
    r = pd.to_numeric(ranks, errors="coerce").to_numpy(dtype=float)
    found = ~np.isnan(r)
    out = {f"recall_at_{k}": float(np.mean(found & (r <= k))) for k in K_VALUES}
    rr = np.where(found & (r <= k_max), 1.0 / np.where(found, r, 1.0), 0.0)
    out[f"mrr_at_{k_max}"] = float(np.mean(rr))
    return out


def summarise(per_query: pd.DataFrame, rank_column: str) -> pd.DataFrame:
    """Per-question metrics plus an overall (micro-averaged) ``ALL`` row."""
    rows = []
    for qid, group in per_query.groupby("question_id", sort=True):
        rows.append({"question_id": qid, "queries": len(group),
                     **metrics_from_ranks(group[rank_column])})
    rows.append({"question_id": "ALL", "queries": len(per_query),
                 **metrics_from_ranks(per_query[rank_column])})
    return pd.DataFrame(rows)
