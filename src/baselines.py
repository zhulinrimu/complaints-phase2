#!/usr/bin/env python3
"""Three simple, credible baselines evaluated once on the frozen test fold.

B0  majority class (constant predictor)
B1  Product -> Issue prior lookup table
B2  bag-of-words multinomial Naive Bayes (log-linear, standard library only)
B3  LLM -- implemented in llm_eval.py and scored from the verbatim response
    cache (results/llm_cache.jsonl) via --backend cache; see report 7.4. Mock
    backend numbers are harness self-tests and are never reported as results.

Outputs: results/table1_baselines.csv / .md, results/baseline_metrics.json,
         results/per_class_recall.csv
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, dump_csv, dump_json, peak_working_set_bytes  # noqa: E402
from clean import load_sample, tokenize  # noqa: E402
from metrics import (accuracy, balanced_accuracy, bootstrap_macro_f1,  # noqa: E402
                     macro_f1, per_class_metrics)
from validate import assert_no_leakage  # noqa: E402

MIN_SUPPORT = 20
OTHER = "__OTHER__"
N_BOOT = 500


def load_folds(root: Path) -> dict[str, list[dict]]:
    rows = load_sample(root / "data" / "derived" / "narrative_sample.csv.gz")
    by_id = {r["id_hash"]: r for r in rows}
    out = {}
    for name in ("train", "val", "test"):
        ids = (root / "data" / "splits" / f"{name}_ids.txt").read_text(
            encoding="utf-8").splitlines()
        out[name] = [by_id[i] for i in ids if i in by_id]
    return out


def label_policy(train: list[dict]) -> tuple[set[str], dict]:
    counts = Counter(r["issue"] for r in train)
    keep = {k for k, v in counts.items() if v >= MIN_SUPPORT}
    collapsed = sum(v for k, v in counts.items() if k not in keep)
    return keep, {
        "min_support": MIN_SUPPORT,
        "kept_classes": len(keep),
        "collapsed_classes": len(counts) - len(keep),
        "collapsed_records": collapsed,
        "collapsed_to": OTHER,
        "reason": "macro-F1 over dozens of classes with <20 training examples is "
                  "dominated by noise; rare classes are reported as one bucket",
    }


def label_of(r: dict, keep: set[str]) -> str:
    return r["issue"] if r["issue"] in keep else OTHER


class BowNaiveBayes:
    """Multinomial NB = a linear model in log-count space; no dependencies."""

    def __init__(self, min_df: int = 2, alpha: float = 1.0):
        self.min_df = min_df
        self.alpha = alpha
        self.logp: dict[str, dict[str, float]] = {}
        self.log_prior: dict[str, float] = {}
        self.default: dict[str, float] = {}
        self.classes: list[str] = []

    def fit(self, texts: list[str], labels: list[str]) -> "BowNaiveBayes":
        df: Counter = Counter()
        per_class: dict[str, Counter] = defaultdict(Counter)
        totals: Counter = Counter()
        for text, lab in zip(texts, labels):
            toks = tokenize(text)
            df.update(set(toks))
            per_class[lab].update(toks)
            totals[lab] += len(toks)
        vocab = {w for w, c in df.items() if c >= self.min_df}
        self.classes = sorted(per_class)
        n = len(labels)
        V = max(1, len(vocab))
        for c in self.classes:
            self.log_prior[c] = math.log((sum(1 for x in labels if x == c) + 1)
                                        / (n + len(self.classes)))
            denom = totals[c] + self.alpha * V
            self.default[c] = math.log(self.alpha / denom)
            self.logp[c] = {w: math.log((cnt + self.alpha) / denom)
                            for w, cnt in per_class[c].items() if w in vocab}
        self.vocab = vocab
        return self

    def predict_one(self, text: str) -> str:
        toks = [w for w in tokenize(text) if w in self.vocab]
        best, best_score = max(
            ((c, self.log_prior[c] + sum(self.logp[c].get(w, self.default[c])
                                         for w in toks)) for c in self.classes),
            key=lambda x: x[1],
        )
        return best

    def predict(self, texts: list[str]) -> list[str]:
        return [self.predict_one(t) for t in texts]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    folds = load_folds(root)
    keep, policy = label_policy(folds["train"])
    assert_no_leakage(["narrative", "product"])  # documented leakage guard

    y_train = [label_of(r, keep) for r in folds["train"]]
    y_test = [label_of(r, keep) for r in folds["test"]]
    x_train = [r["narrative"] for r in folds["train"]]
    x_test = [r["narrative"] for r in folds["test"]]

    majority = Counter(y_train).most_common(1)[0][0]
    product_prior: dict[str, str] = {}
    by_product: dict[str, list[str]] = defaultdict(list)
    for r, y in zip(folds["train"], y_train):
        by_product[r["product"]].append(y)
    for p, ys in by_product.items():
        product_prior[p] = Counter(ys).most_common(1)[0][0]

    results = {}
    per_class_all = {}
    for name, fn in (
        ("B0", lambda: [majority] * len(x_test)),
        ("B1", lambda: [product_prior.get(r["product"], majority)
                        for r in folds["test"]]),
        ("B2", None),
    ):
        t0 = time.perf_counter()
        if name == "B2":
            model = BowNaiveBayes().fit(x_train, y_train)
            pred = model.predict(x_test)
        else:
            pred = fn()
        elapsed = time.perf_counter() - t0
        ci = bootstrap_macro_f1(y_test, pred, n_boot=args.boot, seed=SEED)
        results[name] = {
            "macro_f1": round(macro_f1(y_test, pred), 6),
            "macro_f1_ci95": [ci["ci_low"], ci["ci_high"]],
            "balanced_accuracy": round(balanced_accuracy(y_test, pred), 6),
            "accuracy": round(accuracy(y_test, pred), 6),
            "n_test": len(y_test),
            "n_boot": args.boot,
            "wall_seconds": round(elapsed, 3),
            "peak_working_set_mb": round(peak_working_set_bytes() / 2 ** 20, 1),
        }
        per_class_all[name] = per_class_metrics(y_test, pred)
        print(f"[{name}] macroF1={results[name]['macro_f1']:.4f} "
              f"acc={results[name]['accuracy']:.4f} t={elapsed:.1f}s", flush=True)

    # B3 placeholder.  src/llm_eval.py overwrites this entry with real scores
    # whenever a non-simulated backend (cache replay or live API) is run.
    results["B3"] = {
        "status": "NOT_RUN_IN_THIS_STEP",
        "reason": "computed downstream by src/llm_eval.py on the frozen LLM "
                  "evaluation subset; see results/llm_pilot.json",
        "implementation": "src/llm_eval.py",
        "procedure": "run `python src/llm_eval.py --backend cache` (committed "
                     "response cache) or `--backend api` with LLM_API_KEY; both "
                     "use the same frozen eval IDs, prompt and metrics",
    }

    dump_json({"label_policy": policy, "baselines": results,
               "per_class": per_class_all, "seed": SEED,
               "majority_class": majority,
               "n_train": len(y_train), "n_val": len(folds["val"]),
               "n_test": len(y_test)},
              root / "results" / "baseline_metrics.json")

    rows = []
    for name, m in results.items():
        if m.get("status", "").startswith("NOT_RUN"):
            rows.append([name, "LLM (frozen eval set)", "NOT RUN", "NOT RUN",
                         "NOT RUN", "NOT RUN", "-", "-",
                         "filled by src/llm_eval.py when real responses exist"])
        else:
            rows.append([name, {"B0": "majority class", "B1": "Product->Issue prior",
                                "B2": "bag-of-words Naive Bayes"}[name],
                         m["macro_f1"],
                         f"[{m['macro_f1_ci95'][0]}, {m['macro_f1_ci95'][1]}]",
                         m["balanced_accuracy"], m["accuracy"],
                         m["wall_seconds"], m["peak_working_set_mb"],
                         f"n_test={m['n_test']}, boot={m['n_boot']}"])
    dump_csv(rows, root / "results" / "table1_baselines.csv",
             ["baseline", "definition", "macro_f1", "macro_f1_ci95",
              "balanced_accuracy", "accuracy", "wall_seconds",
              "peak_working_set_mb", "notes"])

    md = ["# Table 1. Initial baseline results (frozen temporal test fold)",
          "",
          f"Target: `Issue`; input: `Consumer complaint narrative`; "
          f"n_train={len(y_train)}, n_val={len(folds['val'])}, n_test={len(y_test)}; "
          f"seed={SEED}; bootstrap={args.boot}.",
          "",
          "| Baseline | Definition | macro-F1 | 95% CI | Balanced acc. | Accuracy | s | MB |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r[:8]) + " |")
    md += ["", "B3 is scored downstream by `src/llm_eval.py` on a frozen "
               "n=400 Issue-stratified subset. This placeholder row is replaced "
               "with real numbers by that script; see section 7.4 of the report.", ""]
    (root / "results" / "table1_baselines.md").write_text("\n".join(md), encoding="utf-8")

    pc_rows = []
    for name, m in per_class_all.items():
        for lab, v in sorted(m.items(), key=lambda kv: -kv[1]["support"]):
            pc_rows.append([name, lab, v["support"], v["recall"], v["precision"], v["f1"]])
    dump_csv(pc_rows, root / "results" / "per_class_recall.csv",
             ["baseline", "issue_label", "support", "recall", "precision", "f1"])

    # Automatically surfaced negative results (no cherry-picking).
    neg = ["# Negative / failed results (auto-generated)", ""]
    if results["B1"]["macro_f1"] <= results["B0"]["macro_f1"]:
        neg.append(f"- B1 (Product prior) did not beat B0 on macro-F1 "
                   f"({results['B1']['macro_f1']} vs {results['B0']['macro_f1']}), "
                   "so Product alone is not a sufficient predictor of Issue. "
                   "Decision: keep B1 only as a transparent reference, do not use "
                   "it as the phase-3 comparator anchor.")
    if results["B2"]["macro_f1"] < 0.5:
        neg.append(f"- B2 macro-F1 is {results['B2']['macro_f1']}: a bag-of-words "
                   "model cannot separate the long tail of Issue labels. "
                   "Decision: phase 3 must report per-class recall, not only an "
                   "aggregate number.")
    neg.append(f"- Constant prediction reaches accuracy "
               f"{results['B0']['accuracy']} on this label distribution while its "
               f"macro-F1 is {results['B0']['macro_f1']}. This is the empirical "
               "reason accuracy is not a primary metric.")
    (root / "results" / "negative_results.md").write_text("\n".join(neg) + "\n",
                                                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
