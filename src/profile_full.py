#!/usr/bin/env python3
"""Single-pass, constant-memory profile of the FULL CFPB complaints.csv.

Produces the evidence used by the Phase-2 report:

  results/full_profile.json          machine-readable full-file statistics
  results/column_profile.csv         per-column missingness / cardinality
  results/monthly_counts.csv         complaints per calendar month
  results/issue_distribution.csv     target-candidate class distribution
  results/product_period.csv         product x time-period cross tab
  results/narrative_coverage.csv     narrative availability by product
  data/derived/narrative_sample.csv.gz   seeded reservoir sample of records
                                         with non-empty narrative + Issue

Duplicate Complaint IDs are counted exactly using 64 disk buckets of 128-bit
hashes, so peak memory stays bounded (this resolves the Phase-1 limitation
"a 31.7M-entry set does not fit comfortably in memory").

Usage:
    python src/profile_full.py --source <complaints.csv.zip> --out-dir . \
                               --sample-size 40000 [--limit 100000]

Core data path: Python standard library only.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    N_COLS, SCHEMA, SEED, CountingStream, dump_csv, dump_json,
    is_masked_zip, normalise_zip, open_csv_stream, parse_iso_date,
    peak_working_set_bytes, sha256_file,
)

# Counters kept as full dictionaries; capped so a pathological high-cardinality
# column cannot exhaust memory.
LOW_CARDINALITY = (
    "Product", "Sub-product", "Issue", "Sub-issue", "Company", "State",
    "Tags", "Submitted via", "Company public response",
    "Company response to consumer", "Timely response?",
)
MAX_DISTINCT = 50000
NARRATIVE_BUCKETS = ((0, 0), (1, 100), (101, 250), (251, 500),
                     (501, 1000), (1001, 2000), (2001, 5000), (5001, 10 ** 9))
ID_BUCKETS = 64
HEAD_N = 100_000


class ExactIdDuper:
    """Exact duplicate count with bounded RAM, using 128-bit ID digests."""

    def __init__(self, tmp_dir: Path):
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.files = [open(self.tmp_dir / f"id_{i:02d}.bin", "wb")
                      for i in range(ID_BUCKETS)]
        self.nonempty = 0

    def add(self, value: str) -> None:
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()
        self.files[digest[0] % ID_BUCKETS].write(digest)
        self.nonempty += 1

    def finish(self) -> dict:
        for f in self.files:
            f.close()
        total = unique = 0
        for i in range(ID_BUCKETS):
            p = self.tmp_dir / f"id_{i:02d}.bin"
            seen = set()
            with p.open("rb") as f:
                while True:
                    block = f.read(1 << 20)
                    if not block:
                        break
                    for j in range(0, len(block), 16):
                        seen.add(block[j:j + 16])
            with p.open("rb") as f:
                n = f.seek(0, 2) // 16
            total += n
            unique += len(seen)
            # Truncate in place instead of unlink(): the 64 intermediate bucket
            # files are deliberately retained (and gitignored) so a run never
            # depends on a bulk-delete operation succeeding.
            with p.open("wb"):
                pass
        return {
            "nonempty_ids": self.nonempty,
            "unique_ids": unique,
            "duplicate_rows": total - unique,
            "duplicate_rate": round((total - unique) / total, 6) if total else 0.0,
            "method": "exact, 64 disk buckets of 128-bit BLAKE2b digests; "
                      "collision probability < 1e-21 at this scale",
        }


def bucket_for_length(n: int) -> str:
    for lo, hi in NARRATIVE_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="complaints.csv or .zip")
    ap.add_argument("--out-dir", default=".", help="repository root")
    ap.add_argument("--sample-size", type=int, default=40_000)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N logical rows (smoke test / documented sample)")
    ap.add_argument("--no-zip-sha", action="store_true")
    args = ap.parse_args()

    root = Path(args.out_dir).resolve()
    source = Path(args.source).resolve()
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print(f"[profile] source={source}", flush=True)
    zip_sha = None if (args.no_zip_sha or source.suffix.lower() != ".zip") \
        else sha256_file(source)
    print(f"[profile] zip sha256={zip_sha}", flush=True)

    counter, text, member = open_csv_stream(source)
    print(f"[profile] member={member}", flush=True)

    idx = {name: i for i, name in enumerate(SCHEMA)}
    missing = [0] * N_COLS
    # A Counter already is the exact distinct-value set; keeping a second set
    # would double per-row work on a 26M-row file.
    truncated = {k: False for k in LOW_CARDINALITY}
    values = {k: Counter() for k in LOW_CARDINALITY}
    zip_kinds = Counter()
    masked_examples = Counter()
    months = Counter()
    month_with_narrative = Counter()
    product_month = defaultdict(Counter)
    product_total = Counter()
    product_with_narrative = Counter()
    narrative_len_buckets = Counter()
    narrative_embedded_newline = 0
    narrative_total_chars = 0
    date_invalid = Counter()
    date_min = {"Date received": None, "Date sent to company": None}
    date_max = {"Date received": None, "Date sent to company": None}
    widths = Counter()
    head = {
        "rows": 0, "narrative_missing": 0, "timely_yes": 0,
        "credit_reporting": 0, "date_min": None, "date_max": None,
        "top_product": Counter(),
    }
    n_logical = 0
    n_with_narrative = 0
    n_with_narrative_and_issue = 0
    n_empty_id = 0
    n_in_progress = 0
    date_descents = 0
    first_date = last_date = prev_date = None
    product_runs = 0
    prev_product = None

    duper = ExactIdDuper(root / "data" / "derived" / "_id_buckets")
    rng = random.Random(SEED)
    reservoir: list[dict] = []
    R = args.sample_size
    n_frame = 0

    reader = csv.reader(text)
    header = next(reader)
    if tuple(h.strip() for h in header) != SCHEMA:
        raise ValueError(f"unexpected header: {header!r}")

    for i, row in enumerate(reader):
        if args.limit and i >= args.limit:
            break
        n_logical += 1
        widths[len(row)] += 1
        if len(row) != N_COLS:
            continue
        for j, v in enumerate(row):
            if not v or v.isspace():
                missing[j] += 1
        product = row[1]

        for k in LOW_CARDINALITY:
            v = row[idx[k]]
            if v:
                c = values[k]
                if len(c) < MAX_DISTINCT or v in c:
                    c[v] += 1
                else:
                    truncated[k] = True

        # dates
        d_recv = parse_iso_date(row[0])
        d_sent = parse_iso_date(row[12])
        if d_recv is None:
            date_invalid["Date received"] += 1
        else:
            ym = d_recv[:7]
            months[ym] += 1
            if first_date is None:
                first_date = d_recv
            last_date = d_recv
            if prev_date is not None and d_recv < prev_date:
                date_descents += 1
            prev_date = d_recv
            if date_min["Date received"] is None or d_recv < date_min["Date received"]:
                date_min["Date received"] = d_recv
            if date_max["Date received"] is None or d_recv > date_max["Date received"]:
                date_max["Date received"] = d_recv
            product_month[product][ym] += 1
        if prev_product is None or product != prev_product:
            product_runs += 1
        prev_product = product
        if d_sent is None:
            date_invalid["Date sent to company"] += 1
        else:
            if date_min["Date sent to company"] is None or d_sent < date_min["Date sent to company"]:
                date_min["Date sent to company"] = d_sent
            if date_max["Date sent to company"] is None or d_sent > date_max["Date sent to company"]:
                date_max["Date sent to company"] = d_sent

        # zip
        zkind = normalise_zip(row[9])
        zip_kinds[zkind] += 1
        if zkind == "masked":
            masked_examples[row[9].strip().upper()] += 1

        # id / duplicates
        cid = row[15]
        if cid.strip():
            duper.add(cid.strip())
        else:
            n_empty_id += 1

        # narrative
        narr = row[5]
        product_total[product] += 1
        if narr.strip():
            n_with_narrative += 1
            L = len(narr)
            narrative_len_buckets[bucket_for_length(L)] += 1
            narrative_total_chars += L
            if "\n" in narr or "\r" in narr:
                narrative_embedded_newline += 1
            product_with_narrative[product] += 1
            if d_recv:
                month_with_narrative[d_recv[:7]] += 1
            issue = row[3].strip()
            if issue:
                n_with_narrative_and_issue += 1
                n_frame += 1
                rec = {
                    "id_hash": hashlib.blake2b(cid.strip().encode(), digest_size=8).hexdigest(),
                    "date": d_recv or "",
                    "product": product,
                    "sub_product": row[2],
                    "issue": issue,
                    "sub_issue": row[4],
                    "company": row[7],
                    "state": row[8],
                    "tags": row[10],
                    "submitted_via": row[11],
                    "timely": row[14],
                    "response": row[13],
                    "narrative": re.sub(r"\s+", " ", narr[:2500]),
                }
                if len(reservoir) < R:
                    reservoir.append(rec)
                else:
                    j = rng.randrange(n_frame)
                    if j < R:
                        reservoir[j] = rec
        if "In progress" in row[13]:
            n_in_progress += 1

        # head-100k comparison block (the exact bug in the Phase-1 sample)
        if i < HEAD_N:
            head["rows"] += 1
            if not row[5].strip():
                head["narrative_missing"] += 1
            if row[14] == "Yes":
                head["timely_yes"] += 1
            if "Credit reporting" in product:
                head["credit_reporting"] += 1
            if d_recv:
                if head["date_min"] is None or d_recv < head["date_min"]:
                    head["date_min"] = d_recv
                if head["date_max"] is None or d_recv > head["date_max"]:
                    head["date_max"] = d_recv
            head["top_product"][product] += 1

        if (i + 1) % 1_000_000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"[profile] {i+1:,} rows  {elapsed:.0f}s  "
                  f"narrative={n_with_narrative:,}  frame={n_frame:,}", flush=True)

    text.close()
    ids = duper.finish()

    # ---- derived tables -------------------------------------------------
    cols = []
    for j, name in enumerate(SCHEMA):
        if name in LOW_CARDINALITY:
            d = len(values[name])
            top = dict(values[name].most_common(15))
        else:
            d, top = None, {}
        cols.append({
            "field": name, "index": j,
            "missing": missing[j],
            "missing_rate": round(missing[j] / n_logical, 6) if n_logical else 0,
            "distinct_observed": d,
            "distinct_truncated": truncated.get(name, False),
            "top_values": top,
        })
    dump_csv([[c["field"], c["index"], c["missing"], c["missing_rate"],
               c["distinct_observed"] if c["distinct_observed"] is not None else "",
               "; ".join(f"{k}={v}" for k, v in list(c["top_values"].items())[:5])]
              for c in cols],
             results / "column_profile.csv",
             ["field", "index", "missing", "missing_rate", "distinct", "top5"])

    month_rows = [[m, months[m], month_with_narrative[m],
                   round(month_with_narrative[m] / months[m], 6) if months[m] else 0]
                  for m in sorted(months)]
    dump_csv(month_rows, results / "monthly_counts.csv",
             ["month", "complaints", "with_narrative", "narrative_rate"])

    issue_rows = [[k, v, round(v / max(1, sum(values["Issue"].values())), 6)]
                  for k, v in values["Issue"].most_common()]
    dump_csv(issue_rows, results / "issue_distribution.csv",
             ["issue", "count", "share"])

    # product x period: two halves of the observed timeline (by sorted months)
    all_months = sorted(months)
    half = all_months[len(all_months) // 2:] if all_months else []
    half_set = set(half)
    pp_rows = []
    for p, total in product_total.most_common(20):
        early = sum(v for m, v in product_month[p].items() if m not in half_set)
        late = sum(v for m, v in product_month[p].items() if m in half_set)
        pp_rows.append([p, total, product_with_narrative[p], early, late])
    dump_csv(pp_rows, results / "product_period.csv",
             ["product", "total", "with_narrative", "early_period", "late_period"])

    cov_rows = [[p, total, product_with_narrative[p],
                 round(product_with_narrative[p] / total, 6) if total else 0]
                for p, total in product_total.most_common(30)]
    dump_csv(cov_rows, results / "narrative_coverage.csv",
             ["product", "total", "with_narrative", "rate"])

    # 70% / 85% cumulative cut points for a temporal split
    cum, cuts = 0, {}
    total_dated = sum(months.values())
    for m in sorted(months):
        cum += months[m]
        for frac in (0.70, 0.85):
            if frac not in cuts and cum >= frac * total_dated:
                cuts[frac] = m
    dump_csv([[f"{f:.2f}", cuts.get(f, "")] for f in (0.70, 0.85)],
             results / "split_cutpoints.csv", ["cumulative_fraction", "month"])
    dump_csv([
        ["first_logical_row_date", first_date or ""],
        ["last_logical_row_date", last_date or ""],
        ["adjacent_date_descents", date_descents],
        ["descent_rate", round(date_descents / max(1, n_logical - 1), 6)],
        ["product_runs", product_runs],
        ["mean_rows_per_product_run", round(n_logical / max(1, product_runs), 1)],
        ["interpretation", "non-zero adjacent date descents + a dominant product in "
                           "the head block => file is NOT chronological and the head "
                           "is not representative; row-index splitting is invalid"],
    ], results / "date_monotonicity.csv", ["metric", "value"])

    # ---- reservoir sample ----------------------------------------------
    sample_path = root / "data" / "derived" / "narrative_sample.csv.gz"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(reservoir[0].keys()) if reservoir else [
        "id_hash", "date", "product", "sub_product", "issue", "sub_issue",
        "company", "state", "tags", "submitted_via", "timely", "response", "narrative",
    ]
    with gzip.open(sample_path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in reservoir:
            w.writerow(rec)

    elapsed = time.perf_counter() - t0
    head["top_product"] = dict(head["top_product"].most_common(8))
    head["narrative_missing_rate"] = round(
        head["narrative_missing"] / head["rows"], 6) if head["rows"] else 0
    head["timely_yes_rate"] = round(head["timely_yes"] / head["rows"], 6) if head["rows"] else 0
    head["credit_reporting_rate"] = round(
        head["credit_reporting"] / head["rows"], 6) if head["rows"] else 0

    profile = {
        "meta": {
            "script": "src/profile_full.py",
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": SEED,
            "seed_usage": "reservoir sampling of narrative-bearing records",
            "limit": args.limit or None,
            "is_full_file": not bool(args.limit),
            "elapsed_seconds": round(elapsed, 2),
            "peak_working_set_bytes": peak_working_set_bytes(),
            "python": sys.version.split()[0],
            "dependencies": "Python standard library only for the data path",
        },
        "file": {
            "source": str(source),
            "source_bytes": source.stat().st_size,
            "zip_sha256": zip_sha,
            "csv_member": member,
            "csv_bytes": counter.bytes_read,
            "csv_sha256": counter.sha256.hexdigest(),
            "physical_lines": counter.newlines,
            "logical_rows": n_logical,
            "row_width_distribution": dict(widths),
            "columns": N_COLS,
            "header": list(SCHEMA),
        },
        "columns": cols,
        "dates": {
            "date_min": date_min, "date_max": date_max,
            "first_logical_row_date": first_date,
            "last_logical_row_date": last_date,
            "invalid_or_missing": dict(date_invalid),
            "adjacent_date_descents": date_descents,
            "product_runs": product_runs,
            "mean_rows_per_product_run": round(n_logical / max(1, product_runs), 1),
            "months": len(months),
            "monthly_counts": dict(sorted(months.items())),
            "monthly_narrative_counts": dict(sorted(month_with_narrative.items())),
            "cumulative_cutpoints": {f"{k:.2f}": v for k, v in cuts.items()},
            "monotonicity_check": "see results/date_monotonicity.csv",
        },
        "narrative": {
            "nonempty": n_with_narrative,
            "nonempty_rate": round(n_with_narrative / n_logical, 6) if n_logical else 0,
            "nonempty_with_issue": n_with_narrative_and_issue,
            "length_buckets": dict(narrative_len_buckets),
            "mean_chars": round(narrative_total_chars / n_with_narrative, 1)
            if n_with_narrative else 0,
            "records_with_embedded_newline": narrative_embedded_newline,
            "sample_frame": n_with_narrative_and_issue,
            "reservoir_size": len(reservoir),
            "reservoir_path": str(sample_path.relative_to(root)),
            "reservoir_truncation_chars": 2500,
        },
        "ids": {**ids, "empty_ids": n_empty_id},
        "zip": {
            "kinds": dict(zip_kinds),
            "masked_examples_top": dict(masked_examples.most_common(12)),
            "policy": "XXXXX / NNNXX are CFPB privacy masks, recorded as 'masked', "
                      "never counted as format errors",
        },
        "labels": {
            "in_progress_rows": n_in_progress,
            "timely_response": dict(values["Timely response?"]),
            "company_response_to_consumer": dict(values["Company response to consumer"]),
            "issue_class_count": len(values["Issue"]),
            "issue_top20": dict(values["Issue"].most_common(20)),
            "product_top20": dict(values["Product"].most_common(20)),
            "state_top15": dict(values["State"].most_common(15)),
            "submitted_via": dict(values["Submitted via"]),
            "tags": dict(values["Tags"]),
        },
        "head_100k": head,
        "product_period_top20": [dict(zip(
            ["product", "total", "with_narrative", "early_period", "late_period"], r))
            for r in pp_rows],
        "narrative_coverage_top30": [dict(zip(
            ["product", "total", "with_narrative", "rate"], r)) for r in cov_rows],
    }
    dump_json(profile, results / "full_profile.json")
    print(f"[profile] DONE rows={n_logical:,} elapsed={elapsed:.1f}s "
          f"csv_sha256={counter.sha256.hexdigest()[:16]}...", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
