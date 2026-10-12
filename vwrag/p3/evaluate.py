"""Graded evaluation of the V1/V2/V3 runs and paired statistics (stage P3-10E).

Protocol (as run in the study):
* Only *judged* candidates are used; unjudged ranks are removed pairwise (not scored as grade 0), preserving the
  relative order of the judged ones.
* Relevant sets are defined from the judged candidates in the system's own top-20: graded = grade >= 1,
  strict = grade 3.  Consequently Recall@20 equals 1 by construction whenever a relevant item exists, and
  nDCG compares each system's ordering with the ideal ordering of *its own* judged items.
* Paired Wilcoxon signed-rank (two-sided), Holm correction over the three system pairs within each metric,
  and a 10,000-resample bootstrap (seed 42) percentile interval for the mean paired difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEYS = ["file_number", "retrieved_file_number"]
METRICS = ["recall_strict_at5", "recall_strict_at10", "recall_strict_at20",
           "recall_graded_at5", "recall_graded_at10", "recall_graded_at20",
           "mrr_strict_at20", "mrr_graded_at20", "ndcg_at5", "ndcg_at10", "ndcg_at20"]
PAIRS = [("InformationWeightedProperty", "PropertyAware"),
         ("UniformProperty", "PropertyAware"),
         ("InformationWeightedProperty", "UniformProperty")]
SYSTEM_NAMES = {"V1": "PropertyAware", "V2": "UniformProperty", "V3": "InformationWeightedProperty"}


def dcg(grades, k: int) -> float:
    vals = np.asarray(grades[:k], dtype=float)
    if len(vals) == 0:
        return 0.0
    return float(np.sum((2 ** vals - 1) / np.log2(np.arange(2, len(vals) + 2))))


def ndcg(grades, k: int) -> float:
    denom = dcg(sorted(grades, reverse=True), k)
    return np.nan if denom == 0 else dcg(grades, k) / denom


def evaluate_query(grades: list) -> dict:
    """Metrics for one query from the grades of its judged candidates, in rank order."""
    nan = np.nan
    if len(grades) == 0:
        return {"judged_count": 0, **{m: (0.0 if m.startswith("mrr") else nan) for m in METRICS}}
    strict_total = sum(g >= 3 for g in grades)
    graded_total = sum(g >= 1 for g in grades)

    def recall(threshold, total, k):
        return nan if total == 0 else sum(g >= threshold for g in grades[:k]) / total

    def mrr(threshold, k):
        for i, g in enumerate(grades[:k], start=1):
            if g >= threshold:
                return 1.0 / i
        return 0.0

    out = {"judged_count": len(grades), "strict_relevant_total": strict_total, "graded_relevant_total": graded_total}
    for k in (5, 10, 20):
        out[f"recall_strict_at{k}"] = recall(3, strict_total, k)
    for k in (5, 10, 20):
        out[f"recall_graded_at{k}"] = recall(1, graded_total, k)
    out["mrr_strict_at20"] = mrr(3, 20)
    out["mrr_graded_at20"] = mrr(1, 20)
    for k in (5, 10, 20):
        out[f"ndcg_at{k}"] = ndcg(grades, k)
    return out


def per_query_metrics(runs: dict[str, pd.DataFrame], relevance: pd.DataFrame) -> pd.DataFrame:
    """``runs`` maps system name -> top-k table (file_number, rank, retrieved_file_number)."""
    rel = relevance[KEYS + ["relevance_grade"]].copy()
    assert not rel.duplicated(KEYS).any(), "duplicate judgments"
    rows = []
    for system, df in runs.items():
        df = df[KEYS + ["rank"]].copy()
        df["rank"] = pd.to_numeric(df["rank"]).astype(int)
        merged = df.merge(rel, on=KEYS, how="left")
        merged = merged[merged["relevance_grade"].notna()].sort_values(["file_number", "rank"], kind="stable")
        graded = {fn: g["relevance_grade"].astype(float).tolist() for fn, g in merged.groupby("file_number", sort=False)}
        for fn in df["file_number"].drop_duplicates():
            rows.append({"retriever": system, "file_number": fn, **evaluate_query(graded.get(fn, []))})
    return pd.DataFrame(rows)


def summarise(query_eval: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, g in query_eval.groupby("retriever"):
        for metric in METRICS:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            rows.append({"retriever": system, "metric": metric, "mean": vals.mean(), "std": vals.std(ddof=1),
                         "median": vals.median(), "n_valid": len(vals), "n_total": len(g)})
    return pd.DataFrame(rows)


def bootstrap_delta(a, b, iterations: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan, np.nan
    boot = d[rng.integers(0, len(d), size=(iterations, len(d)))].mean(axis=1)
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def holm_adjust(pvals) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(np.argsort(pvals)):
        running = max(running, (n - rank) * pvals[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def pairwise_tests(query_eval: pd.DataFrame, pairs=PAIRS, iterations: int = 10000, seed: int = 42) -> pd.DataFrame:
    out = []
    for metric in METRICS:
        recs, raw = [], []
        for exp_name, base_name in pairs:
            a = query_eval[query_eval.retriever == exp_name][["file_number", metric]].rename(columns={metric: "a"})
            b = query_eval[query_eval.retriever == base_name][["file_number", metric]].rename(columns={metric: "b"})
            m = a.merge(b, on="file_number").dropna(subset=["a", "b"])
            if len(m) == 0:
                continue
            diff = m["a"].to_numpy() - m["b"].to_numpy()
            if np.allclose(diff, 0):
                p = 1.0
            else:
                try:
                    p = wilcoxon(diff, alternative="two-sided").pvalue
                except ValueError:
                    p = 1.0
            lo, hi = bootstrap_delta(m["a"], m["b"], iterations, seed)
            recs.append({"metric": metric, "experimental": exp_name, "baseline": base_name,
                         "n_paired_queries": len(m), "experimental_mean": m["a"].mean(), "baseline_mean": m["b"].mean(),
                         "mean_delta": diff.mean(), "wilcoxon_p_raw": p, "bootstrap_ci_lower_95": lo,
                         "bootstrap_ci_upper_95": hi, "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()),
                         "ties": int((diff == 0).sum())})
            raw.append(p)
        if raw:
            for rec, adj in zip(recs, holm_adjust(raw)):
                rec["wilcoxon_p_holm"] = adj
                out.append(rec)
    return pd.DataFrame(out)
