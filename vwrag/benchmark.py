"""Reconstructed V2 benchmark: candidate pools and evaluation queries.

Six retrieval-oriented question types of the 17-question property-valuation
benchmark are evaluated. Each query asks about one valuation file; the query's
single target is that file's ``file_number``. A file enters a question's pool when
it satisfies the eligibility rule below (rules were calibrated against the
counts of an earlier, unrecoverable benchmark; see the paper, Section 5).
"""

from __future__ import annotations

import pandas as pd

from .corpus import NULL_IDS

POOL_ORDER = ["Q1", "Q8", "Q11", "Q14", "Q15", "Q16"]

# Expected size of each reconstructed V2 pool (paper, Section 6).
EXPECTED_V2_COUNTS = {
    "Q1": 4184, "Q8": 3880, "Q11": 1064, "Q14": 4554, "Q15": 4606, "Q16": 2052,
}
EXPECTED_V2_TOTAL = 20340

# Query templates. Q11 deliberately contains no identifier: it asks for *previous*
# reports of "this property/file" and therefore needs historical retrieval.
QUESTION_TEMPLATES = {
    "Q1": "What documents are available for valuation file <FILE_NUMBER>?",
    "Q8": "What supporting documents and evidence are available for file <FILE_NUMBER>?",
    "Q11": "What previous valuation reports are available for this property/file?",
    "Q14": "What documentary/proof evidence supports the valuation of file <FILE_NUMBER>?",
    "Q15": "What photographs/evidence records are available for file <FILE_NUMBER>?",
    "Q16": "Which statutory/property documents are available for file <FILE_NUMBER>?",
}

EARLIER_REPORT_TYPE = "earlier_valuation_report_file"
STATUTORY_TYPES = {
    "property_tax_bill_file",
    "latest_electricity_bill_file",
    "panchayath_approval_file",
}


def build_pools(unified: pd.DataFrame, attachments: pd.DataFrame) -> dict[str, pd.Series]:
    """Return ``{question_id: Series of file_number}`` for the six pools."""
    q11_files = set(
        attachments.loc[attachments["document_name"] == EARLIER_REPORT_TYPE, "file_number"]
    )
    q16_files = set(
        attachments.loc[attachments["document_name"].isin(STATUTORY_TYPES), "file_number"]
    )
    masks = {
        "Q1": unified["attachment_count"] >= 3,
        "Q8": (unified["attachment_count"] >= 3) & (unified["proof_count"] >= 5),
        "Q11": unified["file_number"].isin(q11_files),
        "Q14": unified["proof_count"] >= 5,
        "Q15": unified["proof_count"] >= 1,
        "Q16": unified["file_number"].isin(q16_files),
    }
    return {qid: unified.loc[masks[qid], "file_number"].reset_index(drop=True) for qid in POOL_ORDER}


def check_pool_counts(pools: dict[str, pd.Series]) -> None:
    """Raise if any pool differs from the published V2 size."""
    for qid, expected in EXPECTED_V2_COUNTS.items():
        actual = int(pools[qid].nunique())
        if actual != expected:
            raise AssertionError(f"{qid}: expected {expected} files, found {actual}")


def build_evaluation_set(pools: dict[str, pd.Series]) -> pd.DataFrame:
    """One query per (question type, file) pair, in pool order."""
    parts = []
    for qid in POOL_ORDER:
        files = pools[qid].astype(str).str.strip()
        files = files[~files.isin(NULL_IDS)]
        part = pd.DataFrame({"question_id": qid, "file_number": files.values})
        part["question"] = part["file_number"].map(
            lambda x, q=qid: QUESTION_TEMPLATES[q].replace("<FILE_NUMBER>", str(x))
        )
        parts.append(part)
    evaluation = pd.concat(parts, ignore_index=True)
    if evaluation[["question_id", "file_number"]].duplicated().any():
        raise AssertionError("duplicate question/file pairs in evaluation set")
    return evaluation
