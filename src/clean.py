"""Cleaning / normalisation step of the Phase-2 pipeline.

The single most important invariant: features and labels are returned in two
separate dictionaries so a post-hoc field can never be passed to a model by
accident (see validate.assert_no_leakage).
"""
from __future__ import annotations

import gzip
import csv
import io
import re
from pathlib import Path
from typing import Iterator

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import POST_HOC_FIELDS, SCHEMA, SUBMISSION_TIME_FIELDS, normalise_zip, parse_iso_date  # noqa: E402
from validate import assert_no_leakage  # noqa: E402

WS = re.compile(r"\s+")


def clean_text(value: str, max_chars: int = 2500) -> str:
    return WS.sub(" ", value[:max_chars]).strip()


def clean_row(row: list[str], drop_empty_id: bool = False) -> dict | None:
    """Turn a raw 16-field CSV record into a normalised record, or reject it."""
    if len(row) != 16:
        return None
    cid = row[15].strip()
    if drop_empty_id and not cid:
        return None
    d = parse_iso_date(row[0])
    z = normalise_zip(row[9])
    narr = clean_text(row[5])
    features = {
        "date_received": d or "",
        "product": row[1].strip(),
        "sub_product": row[2].strip(),
        "issue": row[3].strip(),          # target candidate (label, moved below)
        "sub_issue": row[4].strip(),
        "narrative": narr,
        "company": row[7].strip(),
        "state": row[8].strip(),
        "zip_kind": z,
        "zip_clean": row[9].strip() if z == "zip5" else "",
        "tags": row[10].strip(),          # sensitive -> subgroup audit only
        "submitted_via": row[11].strip(),
        "narrative_present": bool(narr),
        "narrative_len": len(narr),
    }
    labels = {
        "company_public_response": row[6].strip(),
        "date_sent_to_company": parse_iso_date(row[12]) or "",
        "company_response_to_consumer": row[13].strip(),
        "timely_response": row[14].strip(),
    }
    # 'issue' is the declared target and must not be an input feature.
    target = features.pop("issue")
    assert_no_leakage(features.keys())
    return {
        "id": cid,
        "features": features,
        "target": target,
        "labels_post_hoc": labels,
    }


def allowed_features() -> tuple[str, ...]:
    """Feature whitelist: submission-time fields minus the target and IDs."""
    return ("date_received", "product", "sub_product", "sub_issue",
            "narrative", "company", "state", "zip_kind", "submitted_via",
            "narrative_present", "narrative_len")


def iter_sample(path: str | Path) -> Iterator[dict]:
    """Iterate the seeded reservoir sample (gzip CSV) as flat dicts."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            yield row


def load_sample(path: str | Path) -> list[dict]:
    return list(iter_sample(path))


def tokenize(text: str) -> list[str]:
    """Minimal, deterministic tokenizer (no external NLP dependency)."""
    return re.findall(r"[a-z][a-z'-]{1,}", text.lower())
