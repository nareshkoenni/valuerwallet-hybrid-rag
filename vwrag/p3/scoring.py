"""The three property scorers of the Paper 3 ablation, vectorised over the corpus.

V1 ``PropertyAware``                 identity score = agreement x coverage; uniform field weights
V2 ``UniformProperty``               matched / available fields; binary survey-token overlap
V3 ``InformationWeightedProperty``   fields weighted by IC = log(N / df); structural-label filtering

All three combine ``0.7 * identity + 0.3 * location`` (frozen configuration, ``config/p3_scoring_v1.json``)
and never return the query document itself.  The implementations are vectorised re-expressions of the
study's per-pair loops and are checked against the saved study rankings (see ``verify_p3.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .normalize import IDENTITY_FIELDS, LOCATION_FIELDS, CONTEXT_FIELDS

IDENTITY_WEIGHT = 0.70
LOCATION_WEIGHT = 0.30
TOP_K = 20

# Values that carry no evidence (missing or malformed).
INVALID_V1 = {"", "nan", "none", "null", "-", "ward number:"}
INVALID_V2 = {"", "-", "nan", "none", "null", "ward number:", "survey number:", "village:",
              "mandal:", "district:", "state:", "pincode:", "building name:", "latitude:"}
# Structural labels that leak into field values when the source line was empty (V3).
STRUCTURAL_LABELS = {"door number:", "ward number:", "plot number:", "survey number:", "village:",
                     "mandal:", "district:", "state:", "pincode:", "building name:", "latitude:",
                     "longitude:", "property type:"}
INVALID_V3 = INVALID_V1 | STRUCTURAL_LABELS


def _clean(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _split_tokens(value: str) -> set[str]:
    """V1/V3 survey tokens: the ' | '-joined normalised representation."""
    return {t.strip() for t in value.split("|") if t.strip()}


def _regex_tokens(value: str) -> set[str]:
    """V2 survey tokens: re-tokenise (sy-no prefix, &, 'and', comma) and drop inner whitespace."""
    s = re.sub(r"^\s*sy\.?\s*no\.?\s*:?\s*", "", value)
    s = s.replace("&", "|")
    s = re.sub(r"\s+and\s+", "|", s)
    s = s.replace(",", "|")
    return {re.sub(r"\s+", "", p.strip()) for p in s.split("|") if re.sub(r"\s+", "", p.strip())}


@dataclass
class _Field:
    values: np.ndarray        # cleaned strings
    valid: np.ndarray         # bool
    codes: np.ndarray         # integer code per distinct value
    tokens: list | None = None
    sizes: np.ndarray | None = None
    inverted: dict | None = None


class PropertyScorer:
    """Holds the corpus-side arrays for one scorer variant."""

    def __init__(self, index: pd.DataFrame, variant: str):
        if variant not in {"V1", "V2", "V3"}:
            raise ValueError(variant)
        self.variant = variant
        self.files = index["file_number"].astype(str).str.strip().to_numpy()
        self.n = len(index)
        self.file_pos = {f: i for i, f in enumerate(self.files)}
        invalid = {"V1": INVALID_V1, "V2": INVALID_V2, "V3": INVALID_V3}[variant]
        tok = _regex_tokens if variant == "V2" else _split_tokens
        self.fields: dict[str, _Field] = {}
        for f in IDENTITY_FIELDS + LOCATION_FIELDS + CONTEXT_FIELDS:
            vals = _clean(index["norm_" + f]).to_numpy(dtype=object)
            valid = np.array([v not in invalid for v in vals])
            codes = pd.factorize(vals)[0]
            fld = _Field(vals, valid, codes)
            if f == "survey_num":
                fld.tokens = [tok(v) if ok else set() for v, ok in zip(vals, valid)]
                fld.sizes = np.array([len(t) for t in fld.tokens])
                inv: dict[str, list[int]] = {}
                for i, ts in enumerate(fld.tokens):
                    for t in ts:
                        inv.setdefault(t, []).append(i)
                fld.inverted = {t: np.array(ix) for t, ix in inv.items()}
            self.fields[f] = fld
        self.tok = tok
        if variant == "V3":   # document frequency of each (valid) normalised value
            self.df = {f: pd.Series(self.fields[f].values[self.fields[f].valid]).value_counts().to_dict()
                       for f in IDENTITY_FIELDS + LOCATION_FIELDS}

    # ------------------------------------------------------------------ query side
    def _query(self, qi: int, override: dict | None = None) -> dict:
        """Query-side value, validity and survey tokens for every field.

        By default the query is the corpus document itself (normalised index values).  ``override`` maps
        field -> raw string and is used to reproduce the study's V2 run, whose queries were the *raw*
        (un-normalised, only lower-cased) metadata fields of the query file."""
        invalid = {"V1": INVALID_V1, "V2": INVALID_V2, "V3": INVALID_V3}[self.variant]
        q = {}
        for f, fld in self.fields.items():
            if override is None:
                value, valid = fld.values[qi], bool(fld.valid[qi])
            else:
                value = str(override.get(f, "") if override.get(f, "") == override.get(f, "") else "").strip().lower()
                valid = value not in invalid
            tokens = self.tok(value) if (f == "survey_num" and valid) else set()
            q[f] = (value, valid, tokens)
        return q

    # ------------------------------------------------------------------ per-field similarity
    def _sim(self, f: str, q: dict):
        """(similarity array, defined mask) of every corpus document against the query.
        ``defined`` is False where the pair carries no evidence for this field."""
        fld = self.fields[f]
        value, valid, A = q[f]
        if not valid:
            return None, None
        if f == "survey_num":
            if not A:
                return None, None
            inter = np.zeros(self.n)
            for t in A:
                ix = fld.inverted.get(t)
                if ix is not None:
                    inter[ix] += 1
            defined = fld.sizes > 0
            union = len(A) + fld.sizes - inter
            sim = np.zeros(self.n)
            sim[defined] = inter[defined] / union[defined]
            if self.variant == "V2":          # binary overlap, not Jaccard
                sim = (inter > 0).astype(float)
            return sim, defined
        sim = (fld.values == value).astype(float)
        return sim, fld.valid

    # ------------------------------------------------------------------ scoring
    def score(self, qi: int, override: dict | None = None) -> dict:
        q = self._query(qi, override)
        if self.variant == "V1":
            return self._score_v1(q)
        if self.variant == "V2":
            return self._score_v2(q)
        return self._score_v3(q, qi)

    def _score_v1(self, q):
        out = {}
        id_num = np.zeros(self.n); id_den = np.zeros(self.n)
        sims = {}
        for f in IDENTITY_FIELDS + LOCATION_FIELDS + CONTEXT_FIELDS:
            sim, d = self._sim(f, q)
            sims[f] = (sim, d)
        for f in IDENTITY_FIELDS:
            sim, d = sims[f]
            if sim is None:
                continue
            id_den += d; id_num += np.where(d, sim, 0.0)
        agree = np.divide(id_num, id_den, out=np.zeros(self.n), where=id_den > 0)
        coverage = id_den / len(IDENTITY_FIELDS)
        id_score = agree * coverage
        loc_num = np.zeros(self.n); loc_den = np.zeros(self.n)
        for f in LOCATION_FIELDS:
            sim, d = sims[f]
            if sim is None:
                continue
            loc_den += d; loc_num += np.where(d, sim, 0.0)
        loc_score = np.divide(loc_num, loc_den, out=np.zeros(self.n), where=loc_den > 0)
        prop = IDENTITY_WEIGHT * id_score + LOCATION_WEIGHT * loc_score
        matches = {f: (np.where(sims[f][1], sims[f][0], 0.0) > 0) if sims[f][0] is not None
                   else np.zeros(self.n, bool) for f in sims}
        id_cnt = sum(matches[f].astype(int) for f in IDENTITY_FIELDS)
        loc_cnt = sum(matches[f].astype(int) for f in LOCATION_FIELDS)
        total = id_cnt + loc_cnt + sum(matches[f].astype(int) for f in CONTEXT_FIELDS)
        return dict(property_score=prop, identity_score=id_score, identity_coverage=coverage,
                    location_score=loc_score, identity_match_count=id_cnt, location_match_count=loc_cnt,
                    matches=matches, sort_keys=[prop, id_score, loc_score, total])

    def _score_v3(self, q, qi):
        n = self.n
        res, matches = {}, {}
        for group, fields in (("identity", IDENTITY_FIELDS), ("location", LOCATION_FIELDS)):
            num = np.zeros(n); den = np.zeros(n); cov_num = np.zeros(n)
            for f in fields:
                matches[f] = np.zeros(n, bool)
                fld = self.fields[f]
                if not q[f][1]:
                    continue
                df = self.df[f].get(q[f][0], 0)
                info = float(np.log(self.n / df)) if df > 0 else 0.0
                if info <= 0:
                    continue
                sim, d = self._sim(f, q)
                if sim is None:
                    continue
                den += np.where(d, info, 0.0)
                num += np.where(d, info * sim, 0.0)
                matches[f] = d & (sim > 0)
                cov_num += np.where(d & (sim > 0), info * sim, 0.0)
            score = np.divide(num, den, out=np.zeros(n), where=den > 0)
            cov = np.divide(cov_num, den, out=np.zeros(n), where=den > 0)
            res[group] = (score, cov)
        prop = IDENTITY_WEIGHT * res["identity"][0] + LOCATION_WEIGHT * res["location"][0]
        id_cnt = sum(matches[f].astype(int) for f in IDENTITY_FIELDS)
        loc_cnt = sum(matches[f].astype(int) for f in LOCATION_FIELDS)
        return dict(property_score=prop, identity_score=res["identity"][0], identity_coverage=res["identity"][1],
                    location_score=res["location"][0], identity_match_count=id_cnt, location_match_count=loc_cnt,
                    matches=matches, sort_keys=[prop, res["identity"][0], res["location"][0], id_cnt, loc_cnt])

    def _score_v2(self, q):
        n = self.n
        matches, avail, hits = {}, {}, {}
        for group, fields in (("identity", IDENTITY_FIELDS), ("location", LOCATION_FIELDS)):
            m = np.zeros(n); a = 0
            for f in fields:
                matches[f] = np.zeros(n, bool)
                if not q[f][1]:
                    continue
                a += 1
                sim, d = self._sim(f, q)
                if sim is None:
                    continue
                matches[f] = d & (sim > 0)
                m += matches[f]
            avail[group] = a; hits[group] = m
        id_score = hits["identity"] / avail["identity"] if avail["identity"] else np.zeros(n)
        loc_score = hits["location"] / avail["location"] if avail["location"] else np.zeros(n)
        prop = 0.7 * id_score + 0.3 * loc_score
        id_cnt = hits["identity"].astype(int); loc_cnt = hits["location"].astype(int)
        return dict(property_score=prop, identity_score=id_score, identity_coverage=id_score,
                    location_score=loc_score, identity_match_count=id_cnt, location_match_count=loc_cnt,
                    matches=matches, sort_keys=[prop, id_cnt, loc_cnt])

    # ------------------------------------------------------------------ ranking
    def rank(self, query_file: str, k: int = TOP_K, override: dict | None = None) -> list[dict]:
        qi = self.file_pos[query_file]
        s = self.score(qi, override)
        keys = [np.asarray(x, dtype=float) for x in s["sort_keys"]]
        if self.variant == "V2":     # ties on counts are broken by ascending file number
            _, tie = np.unique(self.files, return_inverse=True)
            keys = keys + [-tie.astype(float)]
        order = np.lexsort(tuple(-x for x in reversed(keys)))   # descending, stable (corpus order)
        order = order[order != qi][:k]
        rows = []
        for r, j in enumerate(order, start=1):
            rows.append(dict(
                file_number=query_file, rank=r, retrieved_file_number=self.files[j],
                property_score=float(s["property_score"][j]), identity_score=float(s["identity_score"][j]),
                identity_coverage=float(s["identity_coverage"][j]), location_score=float(s["location_score"][j]),
                identity_match_count=int(s["identity_match_count"][j]),
                location_match_count=int(s["location_match_count"][j]),
                identity_matches=" | ".join(f for f in IDENTITY_FIELDS if s["matches"][f][j]),
                location_matches=" | ".join(f for f in LOCATION_FIELDS if s["matches"][f][j])))
        return rows


def retrieve(index: pd.DataFrame, query_files, variant: str, k: int = TOP_K,
             query_overrides: dict | None = None) -> pd.DataFrame:
    """Top-k property retrieval for each query file (all queries are documents of ``index``).

    ``query_overrides`` (file_number -> {field: raw value}) replaces the normalised query values; the
    study's V2 run used raw metadata here (see ``raw_query_values``)."""
    scorer = PropertyScorer(index, variant)
    rows = []
    for qn, qf in enumerate(query_files):
        qf = str(qf).strip()
        for row in scorer.rank(qf, k, None if query_overrides is None else query_overrides[qf]):
            rows.append({"query_index": qn, **row})
    return pd.DataFrame(rows)


def raw_query_values(queries: pd.DataFrame) -> dict:
    """``{file_number: {field: raw value}}`` from the Q11 query table (raw metadata columns, no normalisation)."""
    from .normalize import FIELDS
    out = {}
    for _, row in queries.iterrows():
        out[str(row["file_number"]).strip()] = {f: ("" if pd.isna(row[f]) else str(row[f])) for f in FIELDS}
    return out
