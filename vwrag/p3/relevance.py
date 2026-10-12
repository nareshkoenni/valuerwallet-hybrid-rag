"""Deterministic, rule-based historical-relevance grades for (query file, candidate file) pairs.

Grades: 3 = STRONG/STRICT, 2 = MODERATE, 1 = SUPPORTING, 0 = insufficient.

The grades are computed from the *same* normalised property fields that the scorers use.  They are therefore
not independent of the property representation (see README, "Known limitations").

Two rule variants exist in the study and are both reproduced, because the final judgment pool of Tables 3-4
mixes them (see ``verify_p3.py`` and ``results/paper3/relevance_rule_sensitivity.csv``):

``independent``  stage P3-06I, applied to the 21,280 pairs in the V3 top-20 lists:
                 STRONG  = >= 2 identity-field agreements (location not required)
``protocol``     stage P3-10D, applied to the 19,877 pairs that appear only in the V1/V2 top-20 lists:
                 STRICT  = >= 2 identity-field agreements AND >= 1 location agreement
Both share: MODERATE = >= 1 identity AND >= 2 location agreements;
            SUPPORTING = survey-number token overlap AND >= 1 location agreement.
"""

from __future__ import annotations

import pandas as pd

from .normalize import IDENTITY_FIELDS, LOCATION_FIELDS
from .scoring import INVALID_V3

_PROTOCOL_INVALID = {"", "nan", "none", "null", "-", "ward number:", "survey number:", "village:", "mandal:",
                     "district:", "state:", "building name:", "latitude:", "longitude:"}


def _val(x) -> str:
    return "" if pd.isna(x) else str(x).strip().lower()


def _tokens(x: str, extra_split: bool) -> set[str]:
    if extra_split:
        x = x.replace(",", "|").replace("&", "|")
    return {t.strip() for t in x.split("|") if t.strip()}


def grade_pair(query: dict, cand: dict, rule: str = "independent") -> tuple[int, str, int, int]:
    """Return ``(grade, class, identity_agreements, location_agreements)``.

    ``query`` / ``cand`` map ``norm_<field>`` -> value (rows of the property index)."""
    invalid = INVALID_V3 if rule == "independent" else _PROTOCOL_INVALID
    extra = rule == "protocol"
    id_hits, loc_hits = [], []
    for fields, hits in ((IDENTITY_FIELDS, id_hits), (LOCATION_FIELDS, loc_hits)):
        for f in fields:
            qv, cv = _val(query["norm_" + f]), _val(cand["norm_" + f])
            if qv in invalid or cv in invalid:
                continue
            if f == "survey_num":
                a, b = _tokens(qv, extra), _tokens(cv, extra)
                if a and b and (a & b):
                    hits.append(f)
            elif qv == cv:
                hits.append(f)
    i, l = len(id_hits), len(loc_hits)
    strong = i >= 2 if rule == "independent" else (i >= 2 and l >= 1)
    if strong:
        return 3, "STRONG" if rule == "independent" else "STRICT", i, l
    if i >= 1 and l >= 2:
        return 2, "MODERATE", i, l
    if "survey_num" in id_hits and l >= 1:
        return 1, "SUPPORTING", i, l
    return 0, "INSUFFICIENT", i, l


def grade_pairs(index: pd.DataFrame, pairs: pd.DataFrame, rule: str = "independent",
                queries: pd.DataFrame | None = None) -> pd.DataFrame:
    """Grade every (file_number, retrieved_file_number) row of ``pairs``.

    Candidates are looked up in ``index``.  Queries are looked up in ``queries`` (normalised Q11 metadata, see
    ``build_query_representations``) when given, otherwise in ``index``."""
    lookup = index.set_index("file_number").to_dict("index")
    qlookup = lookup if queries is None else queries.set_index("file_number").to_dict("index")
    out = []
    for q, c in zip(pairs["file_number"].astype(str).str.strip(),
                    pairs["retrieved_file_number"].astype(str).str.strip()):
        g, cls, i, l = grade_pair(qlookup[q], lookup[c], rule)
        out.append((q, c, g, cls, i, l))
    return pd.DataFrame(out, columns=["file_number", "retrieved_file_number", "relevance_grade",
                                      "relevance_class", "identity_match_count", "location_match_count"])
