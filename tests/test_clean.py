import csv
from pathlib import Path

from clean import allowed_features, clean_row, tokenize
from common import open_csv_stream

ROOT = Path(__file__).resolve().parents[1]


def test_quoted_newline_is_one_logical_record():
    path = ROOT / "data" / "sample_complaints.csv"
    counter, text, _ = open_csv_stream(path)
    reader = csv.reader(text)
    header = next(reader)
    rows = list(reader)
    text.close()
    assert len(header) == 16
    assert len(rows) == 11                       # 10 full + 1 deliberately short
    assert counter.newlines > len(rows)   # embedded newlines exist
    assert any("\n" in r[5] for r in rows if len(r) == 16)


def test_clean_row_separates_target_and_post_hoc_labels():
    row = ["2024-01-25T19:56:56.000Z", "Credit reporting", "Credit reporting",
           "Incorrect information on your report", "Account status incorrect",
           "My report is wrong.\nSecond line.", "", "SYNTHETIC BUREAU", "TN",
           "XXXXX", "", "Web", "2024-01-25T19:56:59.000Z",
           "Closed with explanation", "Yes", "123"]
    rec = clean_row(row)
    assert rec["target"] == "Incorrect information on your report"
    assert "issue" not in rec["features"]
    assert "timely_response" not in rec["features"]
    assert rec["labels_post_hoc"]["timely_response"] == "Yes"
    assert rec["features"]["zip_kind"] == "masked"
    assert "\n" not in rec["features"]["narrative"]


def test_width_mismatch_record_is_rejected():
    assert clean_row(["2024-01-01", "Mortgage"]) is None


def test_feature_whitelist_excludes_sensitive_and_post_hoc_fields():
    allowed = allowed_features()
    assert "tags" not in allowed and "zip_clean" not in allowed
    assert "timely_response" not in allowed
    assert "narrative" in allowed


def test_tokenizer_is_deterministic_and_lowercase():
    assert tokenize("Wrong Account, WRONG account's 123!") == [
        "wrong", "account", "wrong", "account's",
    ]
