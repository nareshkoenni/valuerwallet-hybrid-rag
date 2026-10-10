"""Small generic ranking metrics for public examples.

This utility is not the original Paper 2 evaluation implementation.
Ground-truth labels must never be passed to retrieval/routing logic; use them
only after ranking, to compute metrics.
"""
from typing import Iterable, Sequence


def recall_at_k(ranked_items: Sequence[str], relevant_items: Iterable[str], k: int) -> float:
    """Return set-based Recall@k for a single query."""
    relevant = set(relevant_items)
    if not relevant:
        return 0.0
    return len(set(ranked_items[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked_items: Sequence[str], relevant_items: Iterable[str], k: int) -> float:
    """Return reciprocal rank of the first relevant result within the top k."""
    relevant = set(relevant_items)
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0
