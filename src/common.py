"""Shared streaming primitives for the Phase-2 CFPB complaint pipeline.

Only the Python standard library is required for anything that touches the raw
8.9 GB file.  Keeping the data path dependency-free is a deliberate decision:
in Phase 1 the profiler ran on a 548-row head sample; Phase 2 must be able to
run on the complete file in a clean environment.

Author: GTI R&D (Phase 2)
"""
from __future__ import annotations

import ctypes
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Iterator

# Exact 16-column schema of the CFPB Consumer Complaint Database CSV.
SCHEMA = (
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company public response",
    "Company",
    "State",
    "ZIP code",
    "Tags",
    "Submitted via",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
    "Complaint ID",
)
N_COLS = len(SCHEMA)

# Fields that only exist *after* CFPB has forwarded the complaint.  They must
# never appear in a feature matrix, unless one of them is explicitly declared
# as the (retrospective) target.
POST_HOC_FIELDS = (
    "Company public response",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
)

# Fields available at submission time.
SUBMISSION_TIME_FIELDS = tuple(c for c in SCHEMA if c not in POST_HOC_FIELDS)

SEED = 20261107


class CountingStream(io.RawIOBase):
    """Pass-through raw stream that counts bytes/newlines and hashes content.

    Wrapping the ZipExtFile lets a *single* decompression pass produce the CSV
    SHA256, the uncompressed byte count and the physical-line count while
    ``csv.reader`` consumes logical records.  A second 8.9 GB pass is avoided.
    """

    def __init__(self, raw: io.RawIOBase) -> None:
        self.raw = raw
        self.bytes_read = 0
        self.newlines = 0
        self.sha256 = hashlib.sha256()

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        data = self.raw.read(len(b))
        if not data:
            return 0
        n = len(data)
        b[:n] = data
        self.bytes_read += n
        self.newlines += data.count(b"\n")
        self.sha256.update(data)
        return n

    def close(self) -> None:
        try:
            self.raw.close()
        finally:
            super().close()


def open_csv_stream(path: str | Path) -> tuple[CountingStream, io.TextIOWrapper, str]:
    """Open a plain .csv or the first .csv member of a .zip as a text stream.

    ``newline=''`` is mandatory: complaint narratives contain embedded line
    breaks inside quoted fields, so a logical record is not a physical line.
    Returns (counter, text_wrapper, member_name).
    """
    path = Path(path)
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"no .csv member inside {path}")
        member = members[0]
        counter = CountingStream(zf.open(member))
    else:
        member = path.name
        counter = CountingStream(open(path, "rb"))
    buffered = io.BufferedReader(counter, buffer_size=1 << 20)
    text = io.TextIOWrapper(buffered, encoding="utf-8", errors="replace", newline="")
    return counter, text, member


def iter_rows(text: io.TextIOWrapper, expected_header: tuple[str, ...] = SCHEMA):
    """Yield (index, row) logical records, validating the header once."""
    reader = csv.reader(text)
    header = next(reader)
    if tuple(h.strip() for h in header) != expected_header:
        raise ValueError(f"unexpected header: {header!r}")
    for i, row in enumerate(reader):
        yield i, row


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def peak_working_set_bytes() -> int:
    """Peak working set of this process (Windows); 0 if unavailable."""
    try:
        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        # Without explicit argtypes ctypes truncates the 64-bit process handle
        # to a C int on Windows and the call silently fails, returning 0.
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)
        return int(pmc.PeakWorkingSetSize) if ok else 0
    except Exception:
        return 0


def dump_json(obj, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_csv(rows: list[list], path: str | Path, header: list[str] | None = None) -> None:
    import csv as _csv
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        if header:
            w.writerow(header)
        w.writerows(rows)


def is_masked_zip(value: str) -> bool:
    """CFPB masks small-population ZIPs as XXXXX or 123XX; not a format error."""
    v = value.strip().upper()
    if not v:
        return False
    return v == "XXXXX" or (len(v) == 5 and v[:3].isdigit() and v[3:] == "XX")


def normalise_zip(value: str) -> str:
    """Return one of: missing | masked | zip5 | zip5+4 | other."""
    v = value.strip()
    if not v:
        return "missing"
    if is_masked_zip(v):
        return "masked"
    if len(v) == 5 and v.isdigit():
        return "zip5"
    if len(v) == 10 and v[:5].isdigit() and v[5] == "-" and v[6:].isdigit():
        return "zip5+4"
    return "other"


def parse_iso_date(value: str) -> str | None:
    """Return YYYY-MM-DD when the prefix is a valid calendar-looking date."""
    v = value.strip()[:10]
    if len(v) != 10 or v[4] != "-" or v[7] != "-":
        return None
    if not (v[:4].isdigit() and v[5:7].isdigit() and v[8:10].isdigit()):
        return None
    m, d = int(v[5:7]), int(v[8:10])
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return None
    return v
