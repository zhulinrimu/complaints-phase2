"""Evaluation metrics for the Phase-2 baselines and the LLM sub-task.

All metrics are implemented with the standard library so the reported numbers
can be regenerated without installing a scientific stack.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict

ABSTAIN = "ABSTAIN"


def confusion(y_true: list[str], y_pred: list[str]) -> dict[str, Counter]:
    cm: dict[str, Counter] = defaultdict(Counter)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return dict(cm)


def per_class_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, dict]:
    cm = confusion(y_true, y_pred)
    out = {}
    for label in sorted(set(y_true) | set(y_pred)):
        tp = cm.get(label, Counter()).get(label, 0)
        fp = sum(c.get(label, 0) for t, c in cm.items() if t != label)
        fn = sum(v for p, v in cm.get(label, Counter()).items() if p != label)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[label] = {"tp": tp, "fp": fp, "fn": fn,
                      "precision": round(prec, 6), "recall": round(rec, 6),
                      "f1": round(f1, 6), "support": tp + fn}
    return out


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return sum(t == p for t, p in zip(y_true, y_pred)) / max(1, len(y_true))


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Macro-F1 over *all* true+predicted labels (incl. ABSTAIN as its own class)."""
    m = per_class_metrics(y_true, y_pred)
    return sum(v["f1"] for v in m.values()) / max(1, len(m))


def balanced_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    m = per_class_metrics(y_true, y_pred)
    vals = [v["recall"] for k, v in m.items() if v["support"] > 0]
    return sum(vals) / max(1, len(vals))


def bootstrap_macro_f1(y_true: list[str], y_pred: list[str], n_boot: int = 1000,
                       seed: int = 20261107, alpha: float = 0.05) -> dict:
    """Paired bootstrap CI for macro-F1 (percentile method)."""
    rng = random.Random(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        scores.append(macro_f1([y_true[i] for i in idx],
                               [y_pred[i] for i in idx]))
    scores.sort()
    lo = scores[int((alpha / 2) * n_boot)]
    hi = scores[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {
        "point_estimate": round(macro_f1(y_true, y_pred), 6),
        "ci_level": 1 - alpha,
        "ci_low": round(lo, 6), "ci_high": round(hi, 6),
        "n_boot": n_boot, "seed": seed, "method": "paired percentile bootstrap",
    }


def abstention_rate(y_pred: list[str]) -> float:
    return sum(p == ABSTAIN for p in y_pred) / max(1, len(y_pred))


def coverage_weighted_accuracy(y_true: list[str], y_pred: list[str]) -> dict:
    answered = [(t, p) for t, p in zip(y_true, y_pred) if p != ABSTAIN]
    return {
        "coverage": round(1 - abstention_rate(y_pred), 6),
        "accuracy_on_answered": round(accuracy([t for t, _ in answered],
                                               [p for _, p in answered]), 6)
        if answered else 0.0,
        "n_answered": len(answered),
    }


def normalise_for_match(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def groundedness(evidence: str, source_text: str) -> float:
    """Fraction of evidence tokens that are verbatim-supported by the source.

    A conservative lexical check: we split the quoted evidence into content
    words and require the exact normalised phrase to appear in the normalised
    narrative.  Returns 0.0 for empty evidence.
    """
    if not evidence or not evidence.strip():
        return 0.0
    ev = normalise_for_match(evidence)
    src = normalise_for_match(source_text)
    if not ev or not src:
        return 0.0
    if ev in src:
        return 1.0
    # token-level coverage as a graded fallback
    toks = ev.split()
    srcset = set(src.split())
    return round(sum(t in srcset for t in toks) / len(toks), 6)


def output_consistency(repeats: list[list[str]]) -> dict:
    """Agreement across k repeated runs on the same inputs."""
    if not repeats or not repeats[0]:
        return {"k": len(repeats), "mean_pairwise_agreement": 0.0}
    pairs, tot = 0, 0
    for i in range(len(repeats)):
        for j in range(i + 1, len(repeats)):
            tot += sum(a == b for a, b in zip(repeats[i], repeats[j])) / len(repeats[i])
            pairs += 1
    exact = sum(all(r[k] == repeats[0][k] for r in repeats)
                for k in range(len(repeats[0]))) / len(repeats[0])
    return {
        "k": len(repeats),
        "mean_pairwise_agreement": round(tot / max(1, pairs), 6),
        "all_runs_identical_rate": round(exact, 6),
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6))
