import hashlib
import io

from common import (N_COLS, POST_HOC_FIELDS, SCHEMA, CountingStream,
                    is_masked_zip, normalise_zip, parse_iso_date)


def test_schema_is_the_cfpb_16_column_contract():
    assert N_COLS == 16
    assert SCHEMA[0] == "Date received" and SCHEMA[-1] == "Complaint ID"
    assert set(POST_HOC_FIELDS) == {
        "Company public response", "Date sent to company",
        "Company response to consumer", "Timely response?",
    }


def test_privacy_masks_are_detected_not_treated_as_errors():
    assert is_masked_zip("XXXXX")
    assert is_masked_zip("770XX")
    assert is_masked_zip("352xx")
    assert not is_masked_zip("38141")
    assert not is_masked_zip("")


def test_zip_kinds():
    assert normalise_zip("XXXXX") == "masked"
    assert normalise_zip("941XX") == "masked"
    assert normalise_zip("38141") == "zip5"
    assert normalise_zip("94101-1234") == "zip5+4"
    assert normalise_zip("") == "missing"
    assert normalise_zip("abc") == "other"


def test_iso_timestamp_prefix_parsing():
    assert parse_iso_date("2024-01-25T19:56:56.000Z") == "2024-01-25"
    assert parse_iso_date("2026-07-05") == "2026-07-05"
    assert parse_iso_date("not-a-date") is None
    assert parse_iso_date("2024-13-01") is None
    assert parse_iso_date("") is None


def test_counting_stream_counts_bytes_newlines_and_hash():
    payload = b"a,b\n\"c\nd\",e\n"
    cs = CountingStream(io.BytesIO(payload))
    buf = io.BufferedReader(cs, buffer_size=64)
    assert buf.read() == payload
    assert cs.bytes_read == len(payload)
    assert cs.newlines == 3
    assert cs.sha256.hexdigest() == hashlib.sha256(payload).hexdigest()
