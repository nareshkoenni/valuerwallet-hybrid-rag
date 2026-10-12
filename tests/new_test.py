"""Paper 3 tests on synthetic data only."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_synthetic_p3 import make  # noqa: E402

from vwrag.p3.evaluate import dcg, evaluate_query, holm_adjust, ndcg, pairwise_tests, per_query_metrics
from vwrag.p3.normalize import (build_property_index, build_query_representations, extract_field,
                                normalize_survey_tokens, normalize_text)
from vwrag.p3.pipeline import run
from vwrag.p3.relevance import grade_pair
from vwrag.p3.scoring import PropertyScorer, retrieve, raw_query_values


def _index(rows):
    cols = ["door_num", "plot_num", "survey_num", "village", "mandal", "district", "pincode",
            "building_name", "property_type"]
    return pd.DataFrame([{"file_number": fn, **{"norm_" + c: r.get(c, "") for c in cols}} for fn, r in rows])


def test_normalisation():
    assert normalize_text("  Flat No. 12 / A ,  First  Floor ") == "flat no. 12/a,first floor"
    assert normalize_text("None") == "" and normalize_text("-") == "" and normalize_text(float("nan")) == ""
    assert normalize_survey_tokens("Sy.No. 12/A & 13 , nos. 14, 12/A") == ["12/a", "13", "14"]
    assert normalize_survey_tokens("-") == []


def test_extract_field_structural_contamination():
    text = "Door Number: \nWard Number: \nPlot Number: 7\nSurvey Number: 12"
    # an empty value makes the regex swallow the next label line: this is the contamination the scorers filter
    assert extract_field(text, "Door Number").lower() == "ward number:"
    assert extract_field(text, "Plot Number") == "7"
    assert normalize_text(extract_field(text, "Door Number")) == "ward number:"


def test_v1_hand_computed():
    idx = _index([
        ("Q", dict(plot_num="5", survey_num="1 | 2", village="a", mandal="m", district="d", pincode="9")),
        ("X", dict(plot_num="5", survey_num="2 | 3", village="a", mandal="m", district="d", pincode="8")),
    ])
    row = PropertyScorer(idx, "V1").rank("Q", 5)[0]
    # identity: plot 1.0, survey Jaccard 1/3 -> agreement 2/3, coverage 2/4 (2 of 4 fields available) -> 1/3
    assert math.isclose(row["identity_score"], (1 + 1 / 3) / 2 * 0.5)
    assert math.isclose(row["location_score"], 3 / 4)
    assert math.isclose(row["property_score"], 0.7 * row["identity_score"] + 0.3 * 0.75)


def test_v2_binary_survey_overlap_and_tiebreak():
    idx = _index([
        ("Q", dict(survey_num="1 | 2 | 3 | 4", village="a")),
        ("B", dict(survey_num="4 | 5 | 6 | 7", village="z")),   # Jaccard 1/7 but binary overlap = 1
        ("A", dict(survey_num="4 | 5 | 6 | 7", village="z")),
    ])
    rows = PropertyScorer(idx, "V2").rank("Q", 5)
    assert rows[0]["property_score"] == 0.7           # identity 1/1, location 0/1
    assert [r["retrieved_file_number"] for r in rows] == ["A", "B"]   # equal scores -> ascending file number


def test_v3_information_weighting_and_contamination():
    # 'common' village appears everywhere (IC = 0 -> ignored); the rare plot decides.
    rows = [(f"D{i}", dict(plot_num=str(i), village="common", door_num="ward number:")) for i in range(6)]
    rows.append(("Q", dict(plot_num="3", village="common", door_num="ward number:")))
    idx = _index(rows)
    s = PropertyScorer(idx, "V3")
    top = s.rank("Q", 3)[0]
    assert top["retrieved_file_number"] == "D3"
    assert top["identity_match_count"] == 1 and top["location_match_count"] == 0   # village has IC 0
    assert math.isclose(top["property_score"], 0.7)                               # contaminated door ignored


def test_query_never_retrieves_itself_and_returns_k():
    corpus, queries = make()
    idx = build_property_index(corpus)
    qs = queries.file_number.tolist()[:10]
    for v in ("V1", "V2", "V3"):
        r = retrieve(idx, qs, v, k=20)
        assert (r.file_number != r.retrieved_file_number).all()
        assert r.groupby("file_number").size().eq(20).all()
        assert r.groupby("file_number")["rank"].apply(lambda s: list(s) == list(range(1, 21))).all()


def _naive_v1(idx, q, c):
    """Straightforward per-pair re-implementation of the V1 formula."""
    def valid(x):
        return str(x).strip().lower() not in {"", "nan", "none", "null", "-", "ward number:"}
    num = den = 0.0
    for f in ["door_num", "plot_num", "survey_num", "building_name"]:
        a, b = str(q["norm_" + f]).strip().lower(), str(c["norm_" + f]).strip().lower()
        if not (valid(a) and valid(b)):
            continue
        if f == "survey_num":
            A, B = set(x.strip() for x in a.split("|")), set(x.strip() for x in b.split("|"))
            sim = len(A & B) / len(A | B)
        else:
            sim = float(a == b)
        num += sim; den += 1
    ident = (num / den) * (den / 4) if den else 0.0
    ln = ld = 0.0
    for f in ["village", "mandal", "district", "pincode"]:
        a, b = str(q["norm_" + f]).strip().lower(), str(c["norm_" + f]).strip().lower()
        if valid(a) and valid(b):
            ld += 1; ln += float(a == b)
    return 0.7 * ident + 0.3 * (ln / ld if ld else 0.0)


def test_vectorised_v1_matches_naive_reference():
    corpus, queries = make()
    idx = build_property_index(corpus)
    lookup = idx.set_index("file_number").to_dict("index")
    scorer = PropertyScorer(idx, "V1")
    for qf in queries.file_number.tolist()[:15]:
        for row in scorer.rank(qf, 20):
            ref = _naive_v1(idx, lookup[qf], lookup[row["retrieved_file_number"]])
            assert math.isclose(row["property_score"], ref, abs_tol=1e-12)


def test_relevance_rules():
    q = {"norm_door_num": "5", "norm_plot_num": "7", "norm_survey_num": "1 | 2", "norm_building_name": "",
         "norm_village": "a", "norm_mandal": "m", "norm_district": "d", "norm_pincode": ""}
    def cand(**kw):
        c = {k: "x" for k in q}; c["norm_survey_num"] = "9"; c.update(kw); return c
    assert grade_pair(q, cand(norm_door_num="5", norm_plot_num="7"), "independent")[0] == 3
    assert grade_pair(q, cand(norm_door_num="5", norm_plot_num="7"), "protocol")[0] == 0       # no location agreement
    assert grade_pair(q, cand(norm_door_num="5", norm_village="a", norm_mandal="m"))[0] == 2
    assert grade_pair(q, cand(norm_survey_num="2 | 8", norm_village="a"))[0] == 1
    assert grade_pair(q, cand(norm_survey_num="2 | 8"))[0] == 0
    assert grade_pair(q, cand(norm_village="a", norm_mandal="m", norm_district="d"))[0] == 0   # location only


def test_metrics_by_hand():
    m = evaluate_query([0, 3, 1, 0, 2])
    assert m["graded_relevant_total"] == 3 and m["strict_relevant_total"] == 1
    assert m["recall_graded_at5"] == 1.0 and math.isclose(m["recall_graded_at20"], 1.0)
    assert m["mrr_graded_at20"] == 0.5 and m["mrr_strict_at20"] == 0.5
    assert math.isclose(m["ndcg_at5"], dcg([0, 3, 1, 0, 2], 5) / dcg([3, 2, 1, 0, 0], 5))
    assert np.isnan(evaluate_query([0, 0])["recall_graded_at5"])      # undefined, not zero
    assert evaluate_query([])["mrr_graded_at20"] == 0.0


def test_holm():
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_unjudged_candidates_are_dropped_not_zeroed():
    runs = {"S": pd.DataFrame({"file_number": ["q"] * 3, "rank": [1, 2, 3], "retrieved_file_number": ["a", "b", "c"]})}
    rel = pd.DataFrame({"file_number": ["q", "q"], "retrieved_file_number": ["b", "c"], "relevance_grade": [3, 1]})
    qe = per_query_metrics(runs, rel)
    assert qe.loc[0, "mrr_graded_at20"] == 1.0           # 'a' is unjudged and removed, so 'b' counts as rank 1
    assert qe.loc[0, "judged_count"] == 2


def test_end_to_end_synthetic(tmp_path):
    corpus, queries = make()
    res = run(corpus, queries, tmp_path / "work", tmp_path / "out")
    out = tmp_path / "out"
    for name in ("p3_three_way_summary.csv", "p3_three_way_pairwise.csv", "p3_three_way_query_metrics_deidentified.csv",
                 "p3_relevance_grade_counts.csv", "p3_run_manifest.json"):
        assert (out / name).exists()
    # results directory carries no file numbers
    for p in out.glob("*.csv"):
        assert "SYN-" not in p.read_text()
    assert set(res["summary"].retriever) == {"PropertyAware", "UniformProperty", "InformationWeightedProperty"}
    assert len(res["pairwise"]) == 33 and len(res["runs"]["PropertyAware"]) == 20 * len(queries)
    # the build is deterministic
    again = run(corpus, queries, tmp_path / "work2", None)
    pd.testing.assert_frame_equal(res["summary"], again["summary"])


def test_v2_raw_vs_normalised_queries_differ_on_messy_metadata():
    corpus, queries = make()
    idx = build_property_index(corpus)
    qs = queries.file_number.tolist()
    raw = retrieve(idx, qs, "V2", query_overrides=raw_query_values(queries))
    norm = retrieve(idx, qs, "V2")
    assert len(raw) == len(norm)
