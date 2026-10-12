"""Corpus construction: SQLite snapshot -> one retrieval document per valuation file.

The canonical business identifier is ``file_number``. Each valuation file becomes
one text document that combines structured inward-register fields with the
attachment and proof (evidence) inventories linked to the same ``file_number``.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

import pandas as pd

# Identifier strings that are treated as "no identifier".
NULL_IDS = {"", "nan", "None", "NaN"}

INWARD_COLUMNS = [
    "fileNumber", "datasheet", "bank", "valuation_type", "property_type", "scheme",
    "to_whome_it_was_issued", "field_engineer", "report_prepared_by",
    "docs_received_date", "inspection_date", "remarks", "prohibited_status",
    "file_status", "purpose", "plan_approval_authority", "door_num", "ward_num",
    "block_num", "plot_num", "survey_num", "village", "mandal", "district", "state",
    "pincode", "building_name", "inward_date", "customer_details",
    "agent_banker_details", "bank_ref_number", "latitude", "longitude",
    "bill_contact", "letter_engagement",
]

CORPUS_COLUMNS = [
    "file_number", "rag_text", "attachment_count", "document_type_count",
    "proof_count", "proof_type_count", "document_length",
]


def normalize_ids(series: pd.Series) -> pd.Series:
    """Cast to string and strip surrounding whitespace."""
    return series.astype(str).str.strip()


def _drop_null_ids(df: pd.DataFrame, column: str = "file_number") -> pd.DataFrame:
    return df[~df[column].isin(NULL_IDS)].copy()


def _type_summary(df: pd.DataFrame, count_name: str, unique_name: str, types_name: str) -> pd.DataFrame:
    return (
        df.groupby("file_number")
        .agg(
            **{
                count_name: ("document_name", "size"),
                unique_name: ("document_name", "nunique"),
                types_name: (
                    "document_name",
                    lambda x: " | ".join(sorted(set(str(v) for v in x.dropna()))),
                ),
            }
        )
        .reset_index()
    )


def load_tables(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and normalise the inward register, attachment records and proof records."""
    inward = pd.read_sql_query(
        f"SELECT {', '.join(INWARD_COLUMNS)} FROM inward_register", conn
    )
    inward["file_number"] = normalize_ids(inward["fileNumber"])
    inward = _drop_null_ids(inward)
    # One inward record per normalised file number (first occurrence kept).
    inward = inward.drop_duplicates(subset=["file_number"], keep="first").copy()

    attachments = pd.read_sql_query(
        "SELECT fileNumber AS file_number, document_name FROM inward_file_attachments",
        conn,
    )
    attachments["file_number"] = normalize_ids(attachments["file_number"])
    attachments = _drop_null_ids(attachments)

    proofs = pd.read_sql_query(
        "SELECT FILE_NUMBER AS file_number, DOCUMENT_NAME AS document_name FROM PROOFS",
        conn,
    )
    proofs["file_number"] = normalize_ids(proofs["file_number"])
    proofs = _drop_null_ids(proofs)
    return inward, attachments, proofs


def build_unified(
    inward: pd.DataFrame, attachments: pd.DataFrame, proofs: pd.DataFrame
) -> pd.DataFrame:
    """Join inward records with per-file attachment and proof summaries."""
    attachment_summary = _type_summary(
        attachments, "attachment_count", "document_type_count", "document_types"
    )
    proof_summary = _type_summary(proofs, "proof_count", "proof_type_count", "proof_types")

    unified = inward.merge(attachment_summary, on="file_number", how="left")
    unified = unified.merge(proof_summary, on="file_number", how="left")

    for col in ["attachment_count", "document_type_count", "proof_count", "proof_type_count"]:
        unified[col] = unified[col].fillna(0).astype(int)
    for col in ["document_types", "proof_types"]:
        unified[col] = unified[col].fillna("")
    return unified


def clean_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


_VALUATION_FIELDS = [
    ("Datasheet", "datasheet"), ("Bank", "bank"), ("Valuation Type", "valuation_type"),
    ("Property Type", "property_type"), ("Scheme", "scheme"),
    ("Issued To", "to_whome_it_was_issued"), ("Field Engineer", "field_engineer"),
    ("Report Prepared By", "report_prepared_by"), ("File Status", "file_status"),
    ("Purpose", "purpose"), ("Prohibited Status", "prohibited_status"),
    ("Plan Approval Authority", "plan_approval_authority"),
]
_LOCATION_FIELDS = [
    ("Door Number", "door_num"), ("Ward Number", "ward_num"), ("Block Number", "block_num"),
    ("Plot Number", "plot_num"), ("Survey Number", "survey_num"), ("Village", "village"),
    ("Mandal", "mandal"), ("District", "district"), ("State", "state"),
    ("Pincode", "pincode"), ("Building Name", "building_name"),
    ("Latitude", "latitude"), ("Longitude", "longitude"),
]
_DATE_FIELDS = [
    ("Inward Date", "inward_date"), ("Documents Received Date", "docs_received_date"),
    ("Inspection Date", "inspection_date"),
]
_ADDITIONAL_FIELDS = [
    ("Customer Details", "customer_details"), ("Agent/Banker Details", "agent_banker_details"),
    ("Bank Reference Number", "bank_ref_number"), ("Bill Contact", "bill_contact"),
    ("Letter of Engagement", "letter_engagement"), ("Remarks", "remarks"),
]


def _field_lines(row, fields: Iterable[tuple[str, str]]) -> list[str]:
    return [f"{label}: {clean_value(row[col])}" for label, col in fields]


def build_rag_text(row) -> str:
    """Render one valuation file as the text document that is indexed."""
    lines = [f"FILE NUMBER: {clean_value(row['file_number'])}", "", "VALUATION INFORMATION:"]
    lines += _field_lines(row, _VALUATION_FIELDS)
    lines += ["", "PROPERTY / LOCATION:"] + _field_lines(row, _LOCATION_FIELDS)
    lines += ["", "DATES:"] + _field_lines(row, _DATE_FIELDS)
    lines += ["", "ATTACHED DOCUMENT TYPES:", clean_value(row["document_types"])]
    lines += ["", "PROOF / EVIDENCE TYPES:", clean_value(row["proof_types"])]
    lines += [
        "", "EVIDENCE SUMMARY:",
        f"Attachment Count: {row['attachment_count']}",
        f"Document Type Count: {row['document_type_count']}",
        f"Proof Count: {row['proof_count']}",
        f"Proof Type Count: {row['proof_type_count']}",
        "", "ADDITIONAL DETAILS:",
    ]
    lines += _field_lines(row, _ADDITIONAL_FIELDS)
    return "\n".join(lines)


def build_rag_corpus(unified: pd.DataFrame) -> pd.DataFrame:
    """Return the file-level corpus (one row / document per ``file_number``)."""
    corpus = unified.copy()
    corpus["rag_text"] = corpus.apply(build_rag_text, axis=1)
    corpus = corpus[
        ["file_number", "rag_text", "attachment_count", "document_type_count",
         "proof_count", "proof_type_count"]
    ].copy()
    corpus["document_length"] = corpus["rag_text"].str.len()
    assert list(corpus.columns) == CORPUS_COLUMNS
    return corpus


def count_linked(records: pd.DataFrame, inward: pd.DataFrame) -> int:
    """Number of attachment/proof records whose file number exists in the inward register."""
    return int(records["file_number"].isin(set(inward["file_number"])).sum())
