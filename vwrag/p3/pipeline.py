"""End-to-end Paper 3 pipeline: property index -> V1/V2/V3 retrieval -> relevance pool -> evaluation.

Files that contain file numbers or property values (index, rankings, judgments) are written to the *work*
directory, which must stay outside the repository.  Only identifier-free aggregates go to the results directory.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..io_utils import sha256_file, write_json
from .evaluate import KEYS, METRICS, SYSTEM_NAMES, pairwise_tests, per_query_metrics, summarise
from .normalize import FIELDS, build_property_index, build_query_representations
from .relevance import grade_pairs
from .scoring import TOP_K, raw_query_values, retrieve

EXPECTED = {"corpus": 7573, "queries": 1064, "pool": 41157, "independent": 21280, "expansion": 19877}


def make_q11_queries(unified: pd.DataFrame, q11_files) -> pd.DataFrame:
    """Raw property-metadata columns of the Q11 files, in pool order."""
    by_file = unified.set_index("file_number")
    rows = []
    for fn in q11_files:
        rec = {"file_number": fn}
        for f in FIELDS:
            v = by_file.at[fn, f] if f in by_file.columns else ""
            rec[f] = "" if pd.isna(v) else str(v)
        rows.append(rec)
    return pd.DataFrame(rows)


def build_relevance_pool(index: pd.DataFrame, queries_norm: pd.DataFrame, runs: dict[str, pd.DataFrame],
                         grading: str = "study") -> pd.DataFrame:
    """Judged-pair table covering the union of all top-k lists.

    grading="study"       V3's pairs use the ``independent`` rule with queries read from the index; every other
                          pair uses the ``protocol`` rule with queries from the metadata (how the study's pool
                          was assembled).
    grading="independent" / "protocol"  one rule for every pair (sensitivity analysis); queries come from the
                          index (independent) or the metadata (protocol).
    """
    union = pd.concat([r[KEYS] for r in runs.values()]).drop_duplicates().reset_index(drop=True)
    if grading == "independent":
        return grade_pairs(index, union, "independent")
    if grading == "protocol":
        return grade_pairs(index, union, "protocol", queries_norm)
    if grading != "study":
        raise ValueError(grading)
    v3_pairs = runs[SYSTEM_NAMES["V3"]][KEYS].drop_duplicates()
    first = grade_pairs(index, v3_pairs, "independent")
    rest = union.merge(v3_pairs.assign(_in=1), on=KEYS, how="left")
    rest = rest[rest["_in"].isna()].drop(columns="_in")
    second = grade_pairs(index, rest, "protocol", queries_norm)
    return pd.concat([first, second], ignore_index=True)


def run(corpus: pd.DataFrame, queries: pd.DataFrame, workdir: Path, outdir: Path | None = None, *,
        grading: str = "study", v2_queries: str = "raw", strict: bool = False) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    corpus = corpus.copy()
    corpus["file_number"] = corpus["file_number"].astype(str).str.strip()
    queries = queries.copy()
    queries["file_number"] = queries["file_number"].astype(str).str.strip()
    q_files = queries["file_number"].tolist()
    if strict:
        assert len(corpus) == EXPECTED["corpus"] and len(queries) == EXPECTED["queries"]

    index = build_property_index(corpus)
    qrep = build_query_representations(queries)
    index.to_csv(workdir / "p3_corpus_property_index.csv", index=False)
    qrep.to_csv(workdir / "p3_property_representations.csv", index=False)

    overrides = raw_query_values(queries) if v2_queries == "raw" else None
    runs = {
        SYSTEM_NAMES["V1"]: retrieve(index, q_files, "V1"),
        SYSTEM_NAMES["V2"]: retrieve(index, q_files, "V2", query_overrides=overrides),
        SYSTEM_NAMES["V3"]: retrieve(index, q_files, "V3"),
    }
    for v, name in SYSTEM_NAMES.items():
        runs[name].to_csv(workdir / f"p3_{v}_top{TOP_K}.csv", index=False)

    relevance = build_relevance_pool(index, qrep, runs, grading)
    relevance.to_csv(workdir / "p3_relevance_pool.csv", index=False)

    query_eval = per_query_metrics(runs, relevance)
    summary = summarise(query_eval)
    pairwise = pairwise_tests(query_eval)
    pseudo = {fn: f"q{i + 1:04d}" for i, fn in enumerate(q_files)}
    query_eval_public = query_eval.assign(query_id=query_eval["file_number"].map(pseudo)).drop(columns="file_number")
    query_eval_public = query_eval_public[["retriever", "query_id"] + [c for c in query_eval_public.columns
                                                                         if c not in ("retriever", "query_id")]]
    grade_counts = (relevance.groupby("relevance_grade").size().rename("pairs").reset_index())

    result = {"query_eval": query_eval, "summary": summary, "pairwise": pairwise, "relevance": relevance,
              "runs": runs, "index": index}
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        query_eval_public.to_csv(outdir / "p3_three_way_query_metrics_deidentified.csv", index=False)
        summary.to_csv(outdir / "p3_three_way_summary.csv", index=False)
        pairwise.to_csv(outdir / "p3_three_way_pairwise.csv", index=False)
        grade_counts.to_csv(outdir / "p3_relevance_grade_counts.csv", index=False)
        write_json(outdir / "p3_run_manifest.json", {
            "created_at": datetime.now().isoformat(), "grading": grading, "v2_queries": v2_queries,
            "corpus_documents": len(corpus), "queries": len(queries),
            "judged_pairs": len(relevance), "top_k": TOP_K, "metrics": METRICS,
            "workdir_hashes": {p.name: sha256_file(p) for p in sorted(workdir.glob("p3_*.csv"))},
        })
    if strict:
        assert len(relevance) >= 0
    return result
