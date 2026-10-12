"""Synthetic fixtures. No real valuation data is used anywhere in the test-suite."""

import sqlite3
import zlib

import numpy as np
import pytest

from vwrag.corpus import INWARD_COLUMNS


def fake_encode(texts, dim=64):
    """Deterministic hashed bag-of-words encoder (stand-in for a sentence-transformer)."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in str(text).lower().split():
            out[row, zlib.crc32(token.encode()) % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1.0, norms)


@pytest.fixture()
def toy_db(tmp_path):
    path = tmp_path / "toy.sqlite"
    conn = sqlite3.connect(path)
    columns = ", ".join(f'"{c}" TEXT' for c in INWARD_COLUMNS)
    conn.execute(f"CREATE TABLE inward_register ({columns})")
    conn.execute("CREATE TABLE inward_file_attachments (fileNumber TEXT, document_name TEXT)")
    conn.execute("CREATE TABLE PROOFS (FILE_NUMBER TEXT, DOCUMENT_NAME TEXT)")

    files = [f"T-{i}" for i in range(1, 9)] + [" T-9 ", "T-1", "-"]  # padded id, duplicate, bad id
    for fn in files:
        values = {c: "" for c in INWARD_COLUMNS}
        values.update(fileNumber=fn, village="Testville", property_type="Flat")
        conn.execute(
            f"INSERT INTO inward_register VALUES ({', '.join('?' * len(INWARD_COLUMNS))})",
            [values[c] for c in INWARD_COLUMNS],
        )
    attachments = {
        "T-1": ["deed_file", "data_sheet_file", "earlier_valuation_report_file"],
        "T-2": ["deed_file", "property_tax_bill_file", "other_document_file", "data_sheet_file"],
        "T-3": ["deed_file"],
        "T-4": ["deed_file", "data_sheet_file", "panchayath_approval_file"],
        "T-9": ["deed_file", "data_sheet_file", "other_document_file"],
        "UNLINKED-1": ["deed_file"],
    }
    for fn, docs in attachments.items():
        for d in docs:
            conn.execute("INSERT INTO inward_file_attachments VALUES (?, ?)", (fn, d))
    conn.execute("INSERT INTO inward_file_attachments VALUES (NULL, 'deed_file')")
    proofs = {"T-1": 6, "T-2": 1, "T-4": 5, "T-5": 2, "T-9": 7}
    for fn, n in proofs.items():
        for i in range(n):
            conn.execute("INSERT INTO PROOFS VALUES (?, ?)", (fn, f"photo{i + 1}"))
    conn.commit()
    conn.close()
    return path
