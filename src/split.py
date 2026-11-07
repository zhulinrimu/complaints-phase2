#!/usr/bin/env python3
"""Frozen temporal train/validation/test split and the frozen LLM eval set.

Why temporal: the raw CSV is ordered in product blocks, not by time (proved by
results/date_monotonicity.csv), so row-index splitting would place whole
product categories into a single fold.  Deployment also means predicting
future complaints from past ones.

Outputs (all deterministic, seed = 20261107):
    data/splits/{train,val,test}_ids.txt        SHA256-listed record IDs
    data/splits/llm_eval_ids.txt               400 stratified test records
    data/splits/split_manifest.json            cut dates, sizes, overlap audit
    results/table3_splits.csv                  report table 3
    results/llm_eval_manifest.json            LLM eval frame description
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, dump_csv, dump_json  # noqa: E402
from clean import load_sample  # noqa: E402

LLM_EVAL_N = 400


def sha256_text(lines: list[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update((line + "\n").encode("utf-8"))
    return h.hexdigest()


def write_ids(path: Path, ids: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return sha256_text(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--sample", default="data/derived/narrative_sample.csv.gz")
    ap.add_argument("--llm-eval-n", type=int, default=LLM_EVAL_N)
    args = ap.parse_args()
    root = Path(args.root).resolve()

    rows = [r for r in load_sample(root / args.sample)
            if r.get("date") and r.get("issue") and r.get("narrative", "").strip()]
    rows.sort(key=lambda r: (r["date"], r["id_hash"]))
    n = len(rows)
    if n < 1000:
        raise ValueError(f"sample too small for a split: {n}")

    # Cumulative 70/15/15 by *date*, with whole calendar days kept together.
    per_date = Counter(r["date"] for r in rows)
    cum, cut1, cut2 = 0, None, None
    for d in sorted(per_date):
        cum += per_date[d]
        if cut1 is None and cum >= 0.70 * n:
            cut1 = d
        if cut2 is None and cum >= 0.85 * n:
            cut2 = d
            break

    folds = {"train": [], "val": [], "test": []}
    for r in rows:
        fold = "train" if r["date"] < cut1 else ("val" if r["date"] < cut2 else "test")
        folds[fold].append(r)

    split_dir = root / "data" / "splits"
    hashes = {}
    for name, recs in folds.items():
        ids = [r["id_hash"] for r in recs]
        hashes[name] = write_ids(split_dir / f"{name}_ids.txt", ids)

    # Frozen LLM evaluation frame: stratified by Issue over the TEST fold only.
    rng = random.Random(SEED)
    by_label: dict[str, list[dict]] = {}
    for r in folds["test"]:
        by_label.setdefault(r["issue"], []).append(r)
    for v in by_label.values():
        rng.shuffle(v)
    labels = sorted(by_label, key=lambda k: (-len(by_label[k]), k))
    eval_rows: list[dict] = []
    while len(eval_rows) < min(args.llm_eval_n, len(folds["test"])):
        progressed = False
        for lab in labels:
            if by_label[lab] and len(eval_rows) < args.llm_eval_n:
                eval_rows.append(by_label[lab].pop())
                progressed = True
        if not progressed:
            break
    eval_ids = [r["id_hash"] for r in eval_rows]
    eval_sha = write_ids(split_dir / "llm_eval_ids.txt", eval_ids)

    companies = {k: {r["company"] for r in v} for k, v in folds.items()}
    test_from_train = len(companies["test"] & companies["train"]) / max(1, len(companies["test"]))

    manifest = {
        "seed": SEED,
        "sample_file": args.sample,
        "eligible_records": n,
        "eligibility": "non-empty narrative, non-empty Issue, parseable Date received",
        "split_type": "temporal, whole calendar days kept inside one fold",
        "cut_date_train_val": cut1,
        "cut_date_val_test": cut2,
        "forbidden_practices": [
            "row-index splitting on the raw CSV",
            "fitting vocabulary / encoders / priors before splitting",
            "using the test fold for model or prompt selection",
        ],
        "folds": {
            k: {
                "n": len(v),
                "date_min": v[0]["date"] if v else None,
                "date_max": v[-1]["date"] if v else None,
                "distinct_issues": len({r["issue"] for r in v}),
                "distinct_companies": len(companies[k]),
                "narrative_rate_in_source_frame": 1.0,
                "ids_sha256": hashes[k],
                "ids_file": f"data/splits/{k}_ids.txt",
            } for k, v in folds.items()
        },
        "company_overlap": {
            "test_companies": len(companies["test"]),
            "test_companies_also_in_train": len(companies["test"] & companies["train"]),
            "fraction": round(test_from_train, 6),
            "interpretation": "large national bureaus appear in every period; "
                              "group-by-company splitting would break temporal realism, "
                              "so overlap is measured and reported instead",
        },
        "llm_eval": {
            "n": len(eval_ids),
            "frame": "stratified sample of the TEST fold, Issue-stratified",
            "ids_sha256": eval_sha,
            "ids_file": "data/splits/llm_eval_ids.txt",
            "label_coverage": len({r["issue"] for r in eval_rows}),
            "rule": "eval IDs are excluded from few-shot examples and prompt authoring",
        },
    }
    dump_json(manifest, split_dir / "split_manifest.json")
    dump_json(manifest["llm_eval"], root / "results" / "llm_eval_manifest.json")

    table3 = [[k, v["n"], round(v["n"] / n, 4), v["date_min"], v["date_max"],
               v["distinct_issues"], hashes[k][:16]]
              for k, v in manifest["folds"].items()]
    dump_csv(table3, root / "results" / "table3_splits.csv",
             ["fold", "n", "share", "date_min", "date_max", "distinct_issues", "ids_sha256_prefix"])
    print(json.dumps({"n": n, "cut1": cut1, "cut2": cut2,
                      "sizes": {k: len(v) for k, v in folds.items()},
                      "llm_eval_n": len(eval_ids)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
