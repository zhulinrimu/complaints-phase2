"""Data validation rules for the Phase-2 pipeline.

Kept deliberately small and dependency-free so it can be used both on the
full stream and inside unit tests with synthetic rows.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import N_COLS, POST_HOC_FIELDS, SCHEMA, normalise_zip, parse_iso_date  # noqa: E402

ALLOWED_TIMELY = {"Yes", "No", ""}
ALLOWED_TAGS = {"", "Servicemember", "Older American",
                "Servicemember, Older American"}
ALLOWED_VIA = {"Web", "Phone", "Postal mail", "Fax", "Referral", "Email", ""}


def validate_row(row: list[str]) -> list[str]:
    """Return issue codes for one logical CSV record (empty list == valid)."""
    issues: list[str] = []
    if len(row) != N_COLS:
        return ["WIDTH_MISMATCH"]
    for name in ("Date received", "Date sent to company"):
        v = row[SCHEMA.index(name)]
        if v.strip() and parse_iso_date(v) is None:
            issues.append("BAD_DATE:" + name)
    kind = normalise_zip(row[SCHEMA.index("ZIP code")])
    if kind == "other":
        issues.append("ZIP_OTHER_FORMAT")
    # 'masked' is a CFPB privacy mechanism, NOT a data error (Phase-1 fix).
    if not row[SCHEMA.index("Complaint ID")].strip():
        issues.append("EMPTY_COMPLAINT_ID")
    if row[SCHEMA.index("Timely response?")] not in ALLOWED_TIMELY:
        issues.append("TIMELY_OUT_OF_DOMAIN")
    if row[SCHEMA.index("Tags")] not in ALLOWED_TAGS:
        issues.append("TAGS_OUT_OF_DOMAIN")
    if row[SCHEMA.index("Submitted via")] not in ALLOWED_VIA:
        issues.append("VIA_OUT_OF_DOMAIN")
    return issues


class ValidationReport:
    """Streaming accumulator of validation outcomes."""

    def __init__(self) -> None:
        self.rows = 0
        self.valid = 0
        self.codes: Counter[str] = Counter()
        self.examples: dict[str, str] = {}

    def note(self, row: list[str]) -> list[str]:
        self.rows += 1
        codes = validate_row(row)
        if not codes:
            self.valid += 1
        for c in codes:
            self.codes[c] += 1
            self.examples.setdefault(c, "|".join(row[:3])[:120])
        return codes

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "valid": self.valid,
            "valid_rate": round(self.valid / self.rows, 6) if self.rows else 0,
            "issue_codes": dict(self.codes),
            "examples": self.examples,
            "post_hoc_fields_excluded_from_features": list(POST_HOC_FIELDS),
        }


def assert_no_leakage(feature_names: Iterable[str]) -> None:
    """Hard guard used by every training/evaluation entry point."""
    bad = sorted(set(feature_names) & set(POST_HOC_FIELDS))
    if bad:
        raise ValueError(f"leakage: post-hoc fields used as features: {bad}")
