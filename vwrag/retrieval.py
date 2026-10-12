"""Retrieval regimes evaluated in the study.

* ``semantic``                    - dense retrieval over the document embeddings.
* ``semantic_identifier_hidden``  - dense retrieval after the file number is replaced
                                    by the placeholder ``<FILE_NUMBER>`` in the query.
* ``exact_text``                  - exact file-number lookup using the identifier
                                    *extracted from the query text* (no use of labels).
* ``hybrid``                      - hard routing: exact lookup when the query contains a
                                    known file number, otherwise dense retrieval.
* ``exact_label_lookup``          - checks whether the *gold target* exists in the corpus
                                    and assigns rank 1. This reproduces the "Exact
                                    file-number" row reported in the paper's results table,
                                    but it does not read the query text, so it is a
                                    label-lookup diagnostic and not a retrieval method.
"""

from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import summarise

PLACEHOLDER = "<FILE_NUMBER>"

# Template-aware identifier extraction: the identifier is whatever follows the word
# "file" at the end of the question, and it only counts if it is a known file number.
_TEMPLATE_RE = re.compile(r"\bfile\s+(.+?)\s*\?\s*$", re.IGNORECASE)

SYSTEMS = (
    "semantic",
    "semantic_identifier_hidden",
    "exact_text",
    "hybrid",
    "exact_label_lookup",
)


def extract_identifier(question: str, known_ids: set[str]) -> Optional[str]:
    """Return the file number mentioned in ``question`` or ``None``."""
    match = _TEMPLATE_RE.search(str(question))
    if not match:
        return None
    candidate = match.group(1).strip()
    return candidate if candidate in known_ids else None


def mask_identifier(question: str, identifier: Optional[str]) -> str:
    """Replace the identifier (if any) by the placeholder; other text is unchanged."""
    if identifier:
        return str(question).replace(identifier, PLACEHOLDER)
    return str(question)


def semantic_topk(
    query_embeddings: np.ndarray, doc_embeddings: np.ndarray, k: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` document indices and scores (descending) for each query."""
    sims = query_embeddings @ doc_embeddings.T
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[rows, part], axis=1)
    top = part[rows, order]
    return top, sims[rows, top]


def _semantic_ranks(
    questions: Sequence[str],
    targets: Sequence[str],
    doc_ids: np.ndarray,
    doc_embeddings: np.ndarray,
    encode: Callable[[list[str]], np.ndarray],
    k: int,
    batch_size: int,
) -> list[float]:
    """1-based rank of each target within the top-``k`` (NaN if absent).

    Identical query strings are encoded once, which matters for the
    identifier-hidden ablation where all queries of a type share one string.
    """
    unique = list(dict.fromkeys(questions))
    position = {q: i for i, q in enumerate(unique)}
    top_ids = []
    for start in range(0, len(unique), batch_size):
        batch = unique[start:start + batch_size]
        emb = np.asarray(encode(batch), dtype=np.float32)
        top, _ = semantic_topk(emb, doc_embeddings, k)
        top_ids.extend(list(doc_ids[row]) for row in top)
    ranks = []
    for question, target in zip(questions, targets):
        ids = top_ids[position[question]]
        ranks.append(float(ids.index(target) + 1) if target in ids else float("nan"))
    return ranks


def evaluate_systems(
    evaluation: pd.DataFrame,
    doc_ids: Sequence[str],
    doc_embeddings: np.ndarray,
    encode: Callable[[list[str]], np.ndarray],
    k: int = 20,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Evaluate every system and return one row per query with 1-based target ranks."""
    doc_ids = np.asarray([str(x).strip() for x in doc_ids])
    known = set(doc_ids.tolist())
    questions = evaluation["question"].astype(str).tolist()
    targets = evaluation["file_number"].astype(str).str.strip().tolist()
    extracted = [extract_identifier(q, known) for q in questions]

    semantic = _semantic_ranks(questions, targets, doc_ids, doc_embeddings, encode, k, batch_size)
    hidden_questions = [mask_identifier(q, i) for q, i in zip(questions, extracted)]
    hidden = _semantic_ranks(hidden_questions, targets, doc_ids, doc_embeddings, encode, k, batch_size)

    nan = float("nan")
    exact_text = [1.0 if (i is not None and i == t) else nan for i, t in zip(extracted, targets)]
    hybrid = [
        (1.0 if i == t else nan) if i is not None else s
        for i, t, s in zip(extracted, targets, semantic)
    ]
    label_lookup = [1.0 if t in known else nan for t in targets]

    return pd.DataFrame({
        "query_index": np.arange(len(evaluation)),
        "question_id": evaluation["question_id"].to_numpy(),
        "route": ["exact_file_number" if i is not None else "semantic_fallback" for i in extracted],
        "rank_semantic": semantic,
        "rank_semantic_identifier_hidden": hidden,
        "rank_exact_text": exact_text,
        "rank_hybrid": hybrid,
        "rank_exact_label_lookup": label_lookup,
    })


def system_summaries(per_query: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Metrics table for every system (per question type plus overall)."""
    return {
        system: summarise(per_query, f"rank_{system}") for system in SYSTEMS
    }
