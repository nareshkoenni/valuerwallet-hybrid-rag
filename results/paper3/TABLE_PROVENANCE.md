# Provenance of Tables 3 and 4 (Paper 3)

Every number in Table 3 and Table 4 of the manuscript is read from one of the files below.
`python scripts/p3_verify_tables.py` re-checks all 33 Table 3 cells and all 10 Table 4 rows.

| Manuscript item | Artifact | SHA-256 |
|---|---|---|
| Table 3 (macro means) | `p3_three_way_ablation_v2_summary.csv` | `85fdba41da1cc7e5fe46ff9b1328f240c6b6a07be1054ec76ad8cc0442817bb7` |
| Table 4 (Holm p < 0.05 rows) | `p3_three_way_ablation_v2_pairwise.csv` | `0d7b980c17ad5d8b4a1807c56626f2a5b49467771cddb393088179785787645b` |
| Run manifest (V1/V2/V3 definitions, pool sizes) | `p3_three_way_ablation_v2_manifest.json` | `c84c78b6ef3a1c3f1513d3752829be1b83d1f8e45d7263df5c855b952055d239` |
| Appendix A frozen configuration | `p3_property_scoring_v1_config.json` | `c8c92150787062ac6092368c345fd59eb67227e3326a80ef56525e735b7a5228` |

Table 4 as drafted lists 10 rows; the artifact has 11 rows with Holm p < 0.05.
The additional row(s):

* UniformProperty - PropertyAware, recall_graded_at5: delta -0.070077, Holm p 0.000006, 95% CI [-0.102511, -0.038111]

The artifacts are the outputs of the final freeze step (notebook stage P3-10E, `p3_three_way_ablation_v2_*`).
They are byte-identical copies of the study files.  The earlier `p3_three_way_ablation_v1_*`,
`p3_graded_retrieval_evaluation_v1_*` and `p3_corrected_graded_evaluation_v1_*` files are superseded.
