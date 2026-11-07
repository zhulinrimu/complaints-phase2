import pytest

from common import SCHEMA
from validate import ValidationReport, assert_no_leakage, validate_row


def make_row(**overrides):
    row = ["2024-01-25T19:56:56.000Z", "Credit reporting", "Credit reporting",
           "Incorrect information on your report", "Account status incorrect",
           "A short narrative.", "", "SYNTHETIC BUREAU", "TN", "38141", "",
           "Web", "2024-01-25T19:56:59.000Z", "Closed with explanation",
           "Yes", "1234567"]
    for k, v in overrides.items():
        row[SCHEMA.index(k)] = v
    return row


def test_valid_row_has_no_issues():
    assert validate_row(make_row()) == []


def test_width_mismatch_is_fatal_for_that_record():
    assert validate_row(make_row()[:15]) == ["WIDTH_MISMATCH"]


def test_masked_zip_is_not_a_format_error():
    codes = validate_row(make_row(**{"ZIP code": "XXXXX"}))
    assert "ZIP_OTHER_FORMAT" not in codes


def test_bad_date_empty_id_and_out_of_domain_values_are_reported():
    codes = validate_row(make_row(**{
        "Date received": "not-a-date", "Complaint ID": "",
        "Timely response?": "Maybe", "Tags": "Alien",
    }))
    assert "BAD_DATE:Date received" in codes
    assert "EMPTY_COMPLAINT_ID" in codes
    assert "TIMELY_OUT_OF_DOMAIN" in codes
    assert "TAGS_OUT_OF_DOMAIN" in codes


def test_leakage_guard_rejects_post_hoc_fields():
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(["narrative", "Timely response?"])
    assert_no_leakage(["narrative", "Product", "State"])


def test_validation_report_accumulates():
    rep = ValidationReport()
    rep.note(make_row())
    rep.note(make_row(**{"Complaint ID": ""}))
    rep.note(make_row()[:14])
    d = rep.as_dict()
    assert d["rows"] == 3 and d["valid"] == 1
    assert d["issue_codes"]["EMPTY_COMPLAINT_ID"] == 1
    assert d["issue_codes"]["WIDTH_MISMATCH"] == 1
