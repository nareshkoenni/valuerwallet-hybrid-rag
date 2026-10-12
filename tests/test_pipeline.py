import sqlite3

import numpy as np
import pandas as pd

from vwrag.benchmark import QUESTION_TEMPLATES, build_evaluation_set, build_pools
from vwrag.corpus import build_rag_corpus, build_unified, load_tables
from vwrag.metrics import metrics_from_ranks
from vwrag.retrieval import evaluate_systems, extract_identifier, mask_identifier, system_summaries

from conftest import fake_encode


def _unified(db):
    conn = sqlite3.connect(db)
    inward, attachments, proofs = load_tables(conn)
    conn.close()
    return inward, attachments, build_unified(inward, attachments, proofs)


def test_corpus_normalisation(toy_db):
    inward, _, unified = _unified(toy_db)
    # padded id stripped, duplicate "T-1" collapsed; the "-" id is a (degenerate) identifier
    assert "T-9" in set(inward["file_number"])
    assert inward["file_number"].is_unique
    corpus = build_rag_corpus(unified)
    assert corpus["file_number"].is_unique
    assert corpus["rag_text"].str.startswith("FILE NUMBER: ").all()
    assert corpus.loc[corpus.file_number == "T-1", "attachment_count"].iloc[0] == 3


def test_pool_rules(toy_db):
    _, attachments, unified = _unified(toy_db)
    pools = {q: set(s) for q, s in build_pools(unified, attachments).items()}
    assert pools["Q1"] == {"T-1", "T-2", "T-4", "T-9"}        # >= 3 attachments
    assert pools["Q8"] == {"T-1", "T-4", "T-9"}               # >= 3 attachments and >= 5 proofs
    assert pools["Q11"] == {"T-1"}                            # earlier valuation report attached
    assert pools["Q14"] == {"T-1", "T-4", "T-9"}              # >= 5 proofs
    assert pools["Q15"] == {"T-1", "T-2", "T-4", "T-5", "T-9"}  # >= 1 proof
    assert pools["Q16"] == {"T-2", "T-4"}                     # statutory documents attached


def test_identifier_extraction():
    known = {"XX-00001", "ABC-777", "A 12"}
    q = QUESTION_TEMPLATES["Q1"].replace("<FILE_NUMBER>", "XX-00001")
    assert extract_identifier(q, known) == "XX-00001"
    assert extract_identifier("Which statutory/property documents are available for file ABC-777?", known) == "ABC-777"
    assert extract_identifier("What documents are available for valuation file A 12?", known) == "A 12"
    assert extract_identifier("What documents are available for valuation file 999-1?", known) is None
    assert extract_identifier(QUESTION_TEMPLATES["Q11"], known) is None  # no identifier in Q11
    assert mask_identifier(q, "XX-00001").endswith("file <FILE_NUMBER>?")
    assert mask_identifier(QUESTION_TEMPLATES["Q11"], None) == QUESTION_TEMPLATES["Q11"]


def test_metrics_from_ranks():
    m = metrics_from_ranks(pd.Series([1, 2, np.nan, 25]))
    assert m["recall_at_5"] == 0.5 and m["recall_at_20"] == 0.5
    assert abs(m["mrr_at_20"] - (1 + 0.5) / 4) < 1e-12


def test_systems_on_toy_corpus(toy_db):
    _, attachments, unified = _unified(toy_db)
    corpus = build_rag_corpus(unified)
    evaluation = build_evaluation_set(build_pools(unified, attachments))
    doc_emb = fake_encode(corpus["rag_text"].tolist())
    per_query = evaluate_systems(evaluation, corpus["file_number"], doc_emb, fake_encode, k=3, batch_size=4)

    summaries = system_summaries(per_query)
    exact = summaries["exact_text"].set_index("question_id")
    assert exact.loc["Q1", "recall_at_20"] == 1.0
    assert exact.loc["Q11", "recall_at_20"] == 0.0           # no identifier in the text -> no lookup
    label = summaries["exact_label_lookup"].set_index("question_id")
    assert label.loc["Q11", "recall_at_20"] == 1.0           # label lookup never reads the query

    assert (per_query.loc[per_query.question_id == "Q11", "route"] == "semantic_fallback").all()
    assert (per_query.loc[per_query.question_id != "Q11", "route"] == "exact_file_number").all()

    # With the identifier masked, every query of one type is the same string and receives
    # the same ranked list, so at most k distinct targets can ever be hit per question type.
    hidden = per_query[per_query.question_id == "Q1"]["rank_semantic_identifier_hidden"].notna().sum()
    assert hidden <= 3
