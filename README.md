# ValuerWallet Hybrid RAG

Code and aggregate results for the manuscript **"Beyond Dense Retrieval: Information-Need-Aware
Hybrid Retrieval for Evidence-Centric Property-Valuation RAG"** (Koenni and Siddhan; under review).

The study asks whether one dense retriever suffices for heterogeneous property-valuation
information needs, and compares dense retrieval, exact file-number retrieval, and a hard-routed
hybrid on a reconstructed benchmark of **20,340 query–file pairs** over a corpus of **7,573 valuation
files**. This repository holds the code, the benchmark definition, and every *aggregate* result. The
proprietary database and all data derived from it are **not** included (see
[DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)).

## What is in the repository

| Path | Content |
|---|---|
| `vwrag/` | Corpus construction, benchmark reconstruction, retrieval regimes, metrics, CLI |
| `tests/` | Unit tests on a synthetic database (no real data) |
| `vwrag/p3/` | **Paper 3**: property normalisation, V1/V2/V3 scorers, relevance grading, graded evaluation and statistics |
| `scripts/recompute_metrics.py` | Recompute metrics from a saved ranking file (Paper 2) |
| `scripts/make_synthetic_p3.py`, `sample_data/` | Synthetic corpus + Q11 queries for running the Paper 3 pipeline without proprietary data |
| `scripts/p3_verify_tables.py` | Checks Paper 3 Tables 3 and 4 against the released artifacts |
| `benchmark/benchmark_17.csv` | The 17-question benchmark specification (templates only) |
| `results/paper2/` | Metric tables, manifests with SHA-256 fingerprints, and an identifier-free per-query rank table |
| `results/paper3/` | Frozen configuration, Table 3/4 source artifacts with checksums, de-identified per-query metrics, sensitivity analyses |

## Pipeline

```bash
pip install -r requirements.txt
python -m pytest -q                                   # synthetic tests, no data needed

# With authorised access to the research snapshot (kept outside the repository):
python -m vwrag.cli build-corpus    --db PATH/kmvaluation.sqlite --out workdir --strict
python -m vwrag.cli build-benchmark --db PATH/kmvaluation.sqlite --out workdir --strict
python -m vwrag.cli build-index     --corpus workdir/rag_documents.csv --out workdir
python -m vwrag.cli evaluate        --workdir workdir --out results/run1
```

`--strict` asserts the published sizes (7,573 documents; V2 pools Q1=4,184, Q8=3,880, Q11=1,064,
Q14=4,554, Q15=4,606, Q16=2,052). The database is opened read-only.

## Retrieval regimes

| System | Definition |
|---|---|
| `semantic` | `all-MiniLM-L6-v2` (384-d, normalised), cosine top-20 over the 7,573 documents |
| `semantic_identifier_hidden` | Same, after the file number in the query is replaced by `<FILE_NUMBER>` |
| `exact_text` | Exact lookup of the file number **extracted from the query text** |
| `hybrid` | Hard routing: `exact_text` if the query contains a known file number, else `semantic` |
| `exact_label_lookup` | Rank 1 whenever the *gold target* exists in the corpus; does not read the query |

## Results on the frozen V2 benchmark (20,340 queries)

| System | Recall@5 | Recall@10 | Recall@20 | MRR@20 |
|---|---:|---:|---:|---:|
| Dense semantic | 0.1229% | 0.2409% | 0.4425% | 0.1046% |
| Dense, identifier hidden | 0.0344% | 0.0737% | 0.1131% | 0.0218% |
| Exact file-number (`exact_label_lookup`) | 100% | 100% | 100% | 100% |
| Exact file-number (`exact_text`) | 94.7689% | 94.7689% | 94.7689% | 94.7689% |
| Query-aware hybrid | 94.7788% | 94.7984% | 94.8132% | 94.7751% |

Per-question tables are in `results/paper2/`. The hybrid routes 19,276 queries to exact lookup and
1,064 (all Q11) to dense retrieval.

## Reading the results

* **Which queries contain an identifier.** 19,276 of the 20,340 queries (94.77%) name the target
  file number. The 1,064 Q11 ("previous valuation reports for this property/file") queries do not.
* **Two exact-lookup rows.** `exact_label_lookup` reproduces the 100% figure but does not use the
  query text, so it also scores 100% on Q11. `exact_text` extracts the identifier from the query and
  scores 100% on the five identifier-bearing question types and 0% on Q11. The hybrid differs from
  `exact_text` only through the 9 Q11 queries (of 1,064) that dense retrieval places in the top 20.
* **Identifier-hidden ablation.** After masking, every query of a question type is the same string,
  so each type yields a single ranked list and the ablation measures dense retrieval *without any
  target-specific query information*.
* **Supplementary lexical baselines.** The TF-IDF and BM25 numbers reported in the manuscript's
  appendix come from the earlier 20,217-pair benchmark; see
  `results/paper2/appendix_c_lexical_baselines_pre_reconstruction.csv` for their protocol. They are
  not comparable rows of the table above.
* Retrieval only: no generation, faithfulness or citation metrics are evaluated.

## Reproduction record

Checked against the frozen study artifacts:

* `build-corpus` regenerates `rag_documents.csv` **byte-for-byte** (SHA-256 `241cbeff…b721d`).
* `build-benchmark` regenerates all six pools with identical membership and order (20,340 queries).
* Template-based identifier extraction gives the same route split as the study (19,276 / 1,064).
* Metrics recomputed from the saved top-20 rankings match the tables in `results/paper2/`.
* Dense embeddings and rankings were produced during the study (Google Colab, August 2026).
  They were checked by recomputing metrics from the saved rankings, not by re-encoding.

## Paper 3 — property-aware historical retrieval (V1 / V2 / V3)

The Q11 information need ("previous valuation reports for this property/file") has no identifier in the query. Paper 3
retrieves the historical files of the *same property* by comparing a nine-field normalised property representation (door,
plot, survey, building, village, mandal, district, pincode, property type) of the query file with every other file in the
7,573-document corpus, and compares three scorers (`vwrag/p3/scoring.py`; all combine `0.7 * identity + 0.3 * location`
and ignore structural-label contamination such as `ward number:`):

| System | Definition |
|---|---|
| V1 `PropertyAware` | per-field equality, survey Jaccard; identity score = agreement x evidence coverage |
| V2 `UniformProperty` | matched / available fields in each group; survey match = any shared token |
| V3 `InformationWeightedProperty` | fields weighted by `IC = log(N/df)` of the query's value; zero-information values ignored |

```bash
python -m pytest -q                                    # includes Paper 3 tests (synthetic data)

# Demonstration on the synthetic sample (numbers are NOT results of the study):
python -m vwrag.cli p3-run --corpus sample_data/synthetic_rag_documents.csv \
       --queries sample_data/synthetic_q11_queries.csv --workdir /tmp/p3work --out /tmp/p3out

# Study data (authorised access; keep WORKDIR outside the repository):
python -m vwrag.cli p3-queries --db PATH/kmvaluation.sqlite --out WORKDIR --strict
python -m vwrag.cli p3-run --corpus WORKDIR/rag_documents.csv --queries WORKDIR/p3_q11_queries.csv \
       --workdir WORKDIR/p3 --out results/paper3_rerun --strict
python scripts/p3_verify_tables.py
```

`p3-run` writes identifier-bearing intermediates (property index, top-20 lists, judged pairs) to `--workdir` and only
identifier-free aggregates to `--out`.

**Reproduction record (study data).** Rebuilt from the research snapshot, the property index is byte-identical to the
study's (SHA-256 `26c96756…b259`); the V1, V2 and V3 top-20 lists (3 x 21,280 rows) are identical in file numbers, ranks
and scores; all 41,157 relevance grades agree; the per-query metrics, `p3_three_way_ablation_v2_summary.csv` and
`..._pairwise.csv` agree to <= 4e-15. See `results/paper3/TABLE_PROVENANCE.md` for the Table 3/4 sources and checksums.

**Protocol details that affect how the results should be read**

* *Relevance grades are rule-based and derived from the same normalised fields the scorers use* (identity/location
  agreement counts), so they measure agreement with a field-overlap definition of "same property", not independent
  human relevance.
* *The judged pool was assembled in two stages with slightly different rules.* Pairs in V3's top-20 (21,280) use the
  `independent` rule (STRONG = >= 2 identity agreements); the 19,877 pairs found only in V1/V2 lists use the `protocol`
  rule (STRICT additionally needs >= 1 location agreement) and metadata-based query values. `--grading independent|protocol`
  applies one rule to all pairs; `results/paper3/p3_relevance_rule_sensitivity_*.csv` shows the effect.
* *V2's queries are the raw (lower-cased, otherwise un-normalised) metadata of the query file*, whereas V1/V3 read the
  normalised index; `--v2-queries normalised` removes this difference.
* *Unjudged candidates are removed, not scored as 0*, and relevant sets are formed from each system's own judged list.
  Hence Recall@20 = 1 by construction, strict recall is averaged only over the roughly 185–190 (of 1,064) queries with a strict-relevant judged
  candidate, and nDCG measures the ordering of a system's own judged items.
* `results/paper3/p3_property_scoring_v1_config.json` is the frozen V1 configuration (identity x coverage, Jaccard); V2 and V3
  are specified by their implementations here.

## Citation

See [CITATION.cff](CITATION.cff).
