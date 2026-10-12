"""Property-field extraction and normalisation (frozen specification, stage P3-05).

The property index is derived from the ``rag_text`` column of ``rag_documents.csv``.
Representation is normalised; identifier semantics are preserved.
"""

from __future__ import annotations

import re

import pandas as pd

FIELDS = ["door_num", "plot_num", "survey_num", "village", "mandal", "district",
          "pincode", "building_name", "property_type"]
LABELS = {
    "door_num": "Door Number", "plot_num": "Plot Number", "survey_num": "Survey Number",
    "village": "Village", "mandal": "Mandal", "district": "District", "pincode": "Pincode",
    "building_name": "Building Name", "property_type": "Property Type",
}
IDENTITY_FIELDS = ["door_num", "plot_num", "survey_num", "building_name"]
LOCATION_FIELDS = ["village", "mandal", "district", "pincode"]
CONTEXT_FIELDS = ["property_type"]
NORM_COLUMNS = ["norm_" + f for f in FIELDS]

_EMPTY = {"", "nan", "none", "null", "-"}


def normalize_text(value) -> str:
    """Lower-case, map empty markers to '', and tidy spacing around / - , & ( )."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in _EMPTY:
        return ""
    return _tidy(text).strip()


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    for pat, rep in ((r"\s*/\s*", "/"), (r"\s*-\s*", "-"), (r"\s*,\s*", ","),
                     (r"\s*&\s*", "&"), (r"\s*\(\s*", "("), (r"\s*\)\s*", ")")):
        text = re.sub(pat, rep, text)
    return text


def normalize_survey_tokens(value) -> list[str]:
    """Survey-number string -> ordered, de-duplicated list of tokens."""
    if pd.isna(value):
        return []
    text = str(value).strip().lower()
    if text in _EMPTY:
        return []
    text = re.sub(r"^\s*(survey\s*no\.?|survey\s*number|sy\s*no\.?|sy\.?\s*no\.?)\s*:?\s*",
                  "", text, flags=re.IGNORECASE)
    text = _tidy(text)
    tokens = []
    for part in re.split(r"\s*(?:,|&)\s*", text):
        part = re.sub(r"^(?:nos?|numbers?|sy)\.?\s*", "", part.strip(), flags=re.IGNORECASE).strip()
        if part:
            tokens.append(part)
    seen, final = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            final.append(t)
    return final


def extract_field(text, label: str) -> str:
    """Value of the ``Label: value`` line in a RAG document ('' if absent or empty)."""
    if pd.isna(text):
        return ""
    match = re.search(rf"(?im)^\s*{re.escape(label)}\s*:\s*(.*?)\s*$", str(text))
    if not match:
        return ""
    value = match.group(1).strip()
    return "" if value.lower() in _EMPTY else value


def build_property_index(corpus: pd.DataFrame) -> pd.DataFrame:
    """Normalised nine-field property index, one row per corpus document (same order)."""
    rows = []
    for file_number, rag_text in zip(corpus["file_number"], corpus["rag_text"]):
        raw = {f: extract_field(rag_text, LABELS[f]) for f in FIELDS}
        rec = {"file_number": str(file_number).strip()}
        for f in FIELDS:
            if f == "survey_num":
                rec["norm_survey_num"] = " | ".join(normalize_survey_tokens(raw[f]))
            else:
                rec["norm_" + f] = normalize_text(raw[f])
        rows.append(rec)
    return pd.DataFrame(rows, columns=["file_number"] + NORM_COLUMNS)


def build_query_representations(queries: pd.DataFrame) -> pd.DataFrame:
    """Normalised representation of the Q11 queries' *metadata* fields (as opposed to ``rag_text``).

    ``queries`` carries the raw property columns of each Q11 file (door_num ... property_type)."""
    rows = []
    for _, r in queries.iterrows():
        rec = {"file_number": str(r["file_number"]).strip()}
        for f in FIELDS:
            rec["norm_" + f] = (" | ".join(normalize_survey_tokens(r[f])) if f == "survey_num"
                                else normalize_text(r[f]))
        rows.append(rec)
    return pd.DataFrame(rows, columns=["file_number"] + NORM_COLUMNS)
