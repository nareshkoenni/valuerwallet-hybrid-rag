"""Check that Table 3 and Table 4 of the Paper 3 manuscript are exactly what the released artifacts contain.

    python scripts/p3_verify_tables.py [--results results/paper3] [--write-provenance]

Table 3  <- p3_three_way_ablation_v2_summary.csv   (macro means over queries)
Table 4  <- p3_three_way_ablation_v2_pairwise.csv  (rows with Holm p < 0.05)
The expected values below are transcribed from the manuscript (v2 draft, Sections 9.1-9.2).
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

TABLE3 = {  # metric: (V1 PropertyAware, V2 Uniform, V3 IWP)
    "mrr_graded_at20": (0.4911, 0.5719, 0.5158), "mrr_strict_at20": (0.1688, 0.1618, 0.1733),
    "ndcg_at5": (0.8215, 0.8784, 0.8054), "ndcg_at10": (0.8460, 0.8976, 0.8310), "ndcg_at20": (0.8815, 0.9209, 0.8708),
    "recall_graded_at5": (0.7652, 0.7079, 0.7395), "recall_graded_at10": (0.8732, 0.8640, 0.8506),
    "recall_graded_at20": (1.0, 1.0, 1.0), "recall_strict_at5": (0.9582, 0.9414, 0.9676),
    "recall_strict_at10": (0.9848, 0.9716, 0.9902), "recall_strict_at20": (1.0, 1.0, 1.0),
}
SYSTEMS = ("PropertyAware", "UniformProperty", "InformationWeightedProperty")
TABLE4 = [  # experimental, baseline, metric, mean delta, Holm p (6 d.p.), CI low, CI high
    ("InformationWeightedProperty", "PropertyAware", "mrr_graded_at20", 0.024738, 0.000290, 0.011993, 0.037932),
    ("UniformProperty", "PropertyAware", "mrr_graded_at20", 0.080804, 0.0, 0.057109, 0.104610),
    ("InformationWeightedProperty", "UniformProperty", "mrr_graded_at20", -0.056065, 0.000046, -0.078953, -0.032654),
    ("UniformProperty", "PropertyAware", "ndcg_at5", 0.055500, 0.001851, 0.023533, 0.086588),
    ("InformationWeightedProperty", "UniformProperty", "ndcg_at5", -0.082461, 0.000001, -0.115571, -0.050608),
    ("UniformProperty", "PropertyAware", "ndcg_at10", 0.048791, 0.000420, 0.021381, 0.075431),
    ("InformationWeightedProperty", "UniformProperty", "ndcg_at10", -0.072488, 0.000001, -0.101492, -0.044394),
    ("UniformProperty", "PropertyAware", "ndcg_at20", 0.037174, 0.000634, 0.015345, 0.058637),
    ("InformationWeightedProperty", "UniformProperty", "ndcg_at20", -0.055261, 0.000001, -0.078277, -0.032873),
    ("InformationWeightedProperty", "UniformProperty", "mrr_strict_at20", 0.011546, 0.001876, 0.005341, 0.018306),
]
FILES = ["p3_three_way_ablation_v2_summary.csv", "p3_three_way_ablation_v2_pairwise.csv",
         "p3_three_way_ablation_v2_manifest.json", "p3_property_scoring_v1_config.json"]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results" / "paper3"))
    ap.add_argument("--write-provenance", action="store_true")
    a = ap.parse_args()
    R = Path(a.results)
    summary = pd.read_csv(R / FILES[0])
    pairwise = pd.read_csv(R / FILES[1])
    ok = True

    piv = summary.pivot(index="metric", columns="retriever", values="mean")
    for metric, vals in TABLE3.items():
        for system, v in zip(SYSTEMS, vals):
            got = round(float(piv.loc[metric, system]), 4)
            if got != v:
                ok = False
                print(f"TABLE 3 MISMATCH {metric} {system}: manuscript {v} vs artifact {got}")

    sig = pairwise[pairwise["wilcoxon_p_holm"] < 0.05]
    listed = {(e, b, m) for e, b, m, *_ in TABLE4}
    extra = sig[[(r.experimental, r.baseline, r.metric) not in listed for r in sig.itertuples()]]
    for r in extra.itertuples():
        print(f"NOTE: significant row in the artifact but not in the drafted Table 4: {r.experimental} - {r.baseline} "
              f"{r.metric} delta={r.mean_delta:+.6f} Holm p={r.wilcoxon_p_holm:.6f} "
              f"CI=[{r.bootstrap_ci_lower_95:.6f}, {r.bootstrap_ci_upper_95:.6f}]")
    for e, b, m, d, p, lo, hi in TABLE4:
        r = pairwise[(pairwise.experimental == e) & (pairwise.baseline == b) & (pairwise.metric == m)]
        if len(r) != 1:
            ok = False; print("TABLE 4 row missing", e, b, m); continue
        r = r.iloc[0]
        got = (round(r.mean_delta, 6), round(r.wilcoxon_p_holm, 6), round(r.bootstrap_ci_lower_95, 6), round(r.bootstrap_ci_upper_95, 6))
        if any(abs(g - x) > 1.5e-6 for g, x in zip(got, (d, p, lo, hi))):
            ok = False; print("TABLE 4 MISMATCH", e, b, m, got, (d, p, lo, hi))
    print("Table 3: 33 cells checked; Table 4:", len(TABLE4), "listed rows checked ->",
          "ALL MATCH" if ok else "MISMATCH", f"({len(extra)} additional significant row(s) in the artifact)")

    if a.write_provenance and ok:
        lines = ["# Provenance of Tables 3 and 4 (Paper 3)", "",
                 "Every number in Table 3 and Table 4 of the manuscript is read from one of the files below.",
                 "`python scripts/p3_verify_tables.py` re-checks all 33 Table 3 cells and all 10 Table 4 rows.", "",
                 "| Manuscript item | Artifact | SHA-256 |", "|---|---|---|",
                 f"| Table 3 (macro means) | `{FILES[0]}` | `{sha256(R / FILES[0])}` |",
                 f"| Table 4 (Holm p < 0.05 rows) | `{FILES[1]}` | `{sha256(R / FILES[1])}` |",
                 f"| Run manifest (V1/V2/V3 definitions, pool sizes) | `{FILES[2]}` | `{sha256(R / FILES[2])}` |",
                 f"| Appendix A frozen configuration | `{FILES[3]}` | `{sha256(R / FILES[3])}` |", "",
                 f"Table 4 as drafted lists {len(TABLE4)} rows; the artifact has {len(sig)} rows with Holm p < 0.05.",
                 "The additional row(s):", ""] + [
                     f"* {r.experimental} - {r.baseline}, {r.metric}: delta {r.mean_delta:+.6f}, Holm p {r.wilcoxon_p_holm:.6f}, "
                     f"95% CI [{r.bootstrap_ci_lower_95:.6f}, {r.bootstrap_ci_upper_95:.6f}]" for r in extra.itertuples()] + [
                 "", "The artifacts are the outputs of the final freeze step (notebook stage P3-10E, `p3_three_way_ablation_v2_*`).",
                 "They are byte-identical copies of the study files.  The earlier `p3_three_way_ablation_v1_*`,",
                 "`p3_graded_retrieval_evaluation_v1_*` and `p3_corrected_graded_evaluation_v1_*` files are superseded."]
        (R / "TABLE_PROVENANCE.md").write_text("\n".join(lines) + "\n")
        print("wrote", R / "TABLE_PROVENANCE.md")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
