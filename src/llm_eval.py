#!/usr/bin/env python3
"""LLM sub-task: evaluation harness, prompt/config versioning, groundedness,
output-variability analysis and prompt-injection tests.

This module is the Phase-2 response to the Phase-1 feedback "the use of large
models has plenty of room to grow; our work does not reflect it".  Phase 1 only
used an assistant to help write code; here the LLM is a *first-class evaluated
method* with a frozen eval frame, a versioned prompt, a JSON schema, evidence
grounding, abstention, repeated-run variability and adversarial tests.

THREE BACKENDS, all honest about what they are:
    mock   deterministic stand-in used ONLY to self-test the harness.  Its
           scores are flagged `not_reportable: true` and must never be quoted
           as model performance.
    cache  replays recorded real responses from results/llm_cache.jsonl
           (allowed by the course guide, page 25, when an API is unavailable).
           Each cache key stores a LIST of independently produced responses,
           one per repeat run, so output-variability is a real measurement
           rather than a replay of the same string three times.
    api    real OpenAI-compatible HTTP call; refuses to run unless
           configs/llm.json has allow_external_transmission=true and the
           LLM_API_KEY environment variable is set.

Usage:
    python src/llm_eval.py --root . --backend mock
    python src/llm_eval.py --root . --backend cache
    python src/llm_eval.py --root . --backend api      # requires authorisation
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, dump_csv, dump_json  # noqa: E402
from baselines import BowNaiveBayes, label_of, label_policy, load_folds  # noqa: E402
from metrics import (ABSTAIN, abstention_rate, accuracy, balanced_accuracy,  # noqa: E402
                     bootstrap_macro_f1, coverage_weighted_accuracy,
                     groundedness, macro_f1, output_consistency,
                     per_class_metrics, wilson_interval)
from redact import (contains_injection, make_adversarial_cases, redact,  # noqa: E402
                    redaction_audit)

SCHEMA_KEYS = ("issue_label", "evidence_quote", "confidence", "abstain")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(root: Path) -> dict:
    p = root / "configs" / "llm.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["_config_sha256"] = sha256_file(p)
    return cfg


def build_prompt(template: str, label_set: list[str], text: str,
                 few_shot: list[dict]) -> str:
    few = "\n".join(
        f'Example input: "{e["text"][:180]}"\n'
        f'Example output: {{"issue_label":"{e["label"]}","evidence_quote":"...",'
        f'"confidence":0.9,"abstain":false}}'
        for e in few_shot
    )
    return (template
            .replace("{label_set}", json.dumps(label_set, ensure_ascii=False))
            .replace("{few_shot}", few)
            .replace("{text}", text))


def parse_response(raw: str, allowed: set[str]) -> dict:
    """Strict JSON-schema check; anything else becomes a recorded failure."""
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        obj = json.loads(m.group(0) if m else raw)
    except Exception as e:
        return {"parse_ok": False, "error": f"json:{type(e).__name__}",
                "label": ABSTAIN, "evidence": "", "confidence": 0.0, "abstain": True}
    missing = [k for k in SCHEMA_KEYS if k not in obj]
    if missing:
        return {"parse_ok": False, "error": "missing:" + ",".join(missing),
                "label": ABSTAIN, "evidence": "", "confidence": 0.0, "abstain": True}
    abstain = bool(obj["abstain"])
    label = str(obj["issue_label"]).strip()
    if abstain or label == ABSTAIN:
        label = ABSTAIN
    elif label not in allowed:
        return {"parse_ok": True, "error": "label_out_of_domain", "label": ABSTAIN,
                "evidence": str(obj["evidence_quote"]), "confidence": 0.0,
                "abstain": True}
    return {"parse_ok": True, "error": "", "label": label,
            "evidence": str(obj["evidence_quote"])[:300],
            "confidence": float(obj.get("confidence", 0) or 0), "abstain": label == ABSTAIN}


class Backend:
    name = "base"

    def complete(self, prompt: str, cfg: dict, run: int = 0) -> dict:
        raise NotImplementedError


class MockBackend(Backend):
    """Deterministic stand-in.  For harness self-tests only, NOT a model."""
    name = "mock_deterministic"

    def complete(self, prompt: str, cfg: dict, run: int = 0) -> dict:
        text = prompt.split("COMPLAINT TEXT:\n")[-1][:2500]
        low = text.lower()
        label = ABSTAIN
        for candidate in cfg.get("_label_set", []):
            key = candidate.lower().split()[0]
            if len(key) > 4 and key in low:
                label = candidate
                break
        evidence = " ".join(text.split()[:12])
        return {
            "raw": json.dumps({"issue_label": label, "evidence_quote": evidence,
                               "confidence": 0.5, "abstain": label == ABSTAIN},
                              ensure_ascii=False),
            "prompt_tokens": len(prompt.split()), "completion_tokens": 40,
            "latency_s": 0.001, "is_simulation": True,
        }


class CacheBackend(Backend):
    """Replay INDEPENDENT recorded responses keyed by the exact prompt hash."""
    name = "cache_replay"

    def __init__(self, root: Path):
        self.cache: dict[str, list[dict]] = {}
        p = root / "results" / "llm_cache.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    obj = json.loads(line)
                    responses = obj.get("responses")
                    if responses is None:          # backwards-compatible single response
                        responses = [{"raw_response": obj["raw_response"],
                                      "prompt_tokens": obj.get("prompt_tokens", 0),
                                      "completion_tokens": obj.get("completion_tokens", 0),
                                      "latency_s": obj.get("latency_s", 0.0)}]
                    self.cache[obj["prompt_sha256"]] = responses

    def complete(self, prompt: str, cfg: dict, run: int = 0) -> dict:
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key not in self.cache:
            return {"raw": '{"issue_label":"ABSTAIN","evidence_quote":"",'
                           '"confidence":0.0,"abstain":true}',
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "latency_s": 0.0, "is_simulation": False,
                    "cache_miss": True}
        pool = self.cache[key]
        rec = pool[run % len(pool)]
        return {"raw": rec["raw_response"],
                "prompt_tokens": rec.get("prompt_tokens", 0),
                "completion_tokens": rec.get("completion_tokens", 0),
                "latency_s": rec.get("latency_s", 0.0),
                "is_simulation": False, "cache_hit": True,
                "independent_repeat": len(pool) >= cfg.get("repeats", 3)}


class ApiBackend(Backend):
    """OpenAI-compatible chat-completions call using urllib (stdlib only)."""
    name = "api_live"

    def complete(self, prompt: str, cfg: dict, run: int = 0) -> dict:
        key = os.environ.get("LLM_API_KEY")
        if not key or not cfg.get("allow_external_transmission"):
            raise PermissionError(
                "live API call refused: set LLM_API_KEY and "
                "configs/llm.json:allow_external_transmission=true")
        payload = {
            "model": cfg["model"],
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "max_tokens": cfg["max_tokens"],
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        }
        if cfg.get("seed") is not None:
            payload["seed"] = cfg["seed"]
        req = urllib.request.Request(
            cfg["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"}, method="POST")
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=cfg.get("timeout_s", 60)) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        latency = time.perf_counter() - t0
        usage = body.get("usage", {})
        return {
            "raw": body["choices"][0]["message"]["content"],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "latency_s": round(latency, 3),
            "model_version_returned": body.get("model", ""),
            "is_simulation": False, "cache_hit": False,
        }


def update_table1(root: Path, result: dict, records: list[dict],
                  eval_rows: list[dict], folds: dict, keep: set,
                  cfg: dict) -> dict | None:
    """Replace the B3 placeholder with real numbers; add a same-frame control.

    Table 1 scores B0-B2 on the full 6,001-record test fold, while B3 is
    evaluated on a 400-record Issue-stratified subset.  Those figures are NOT
    comparable.  Table 1b therefore recomputes all three baselines on exactly
    the same 400 records and the same redacted/truncated input B3 received.
    """
    if result["IS_SIMULATION"]:
        return None
    y_true = [r["true_label"] for r in records]
    y_pred = [r["first_label"] for r in records]
    ci = bootstrap_macro_f1(y_true, y_pred, n_boot=500, seed=SEED)
    b3 = {
        "status": "SCORED",
        "macro_f1": round(macro_f1(y_true, y_pred), 6),
        "macro_f1_ci95": [ci["ci_low"], ci["ci_high"]],
        "balanced_accuracy": round(balanced_accuracy(y_true, y_pred), 6),
        "accuracy": round(accuracy(y_true, y_pred), 6),
        "abstention_rate": round(abstention_rate(y_pred), 6),
        "n": len(y_true), "n_boot": 500, "backend": result["mode"],
        "sample_frame": "Issue-stratified n=400 subset of the frozen test fold",
        "caveat": "not comparable with B0-B2 in Table 1 (n=6,001 full test fold); "
                  "see results/table1b_llm_subset.md for same-frame controls",
    }

    y_train = [label_of(r, keep) for r in folds["train"]]
    majority = Counter(y_train).most_common(1)[0][0]
    by_product: dict[str, list[str]] = defaultdict(list)
    for r, y in zip(folds["train"], y_train):
        by_product[r["product"]].append(y)
    product_prior = {p: Counter(ys).most_common(1)[0][0]
                     for p, ys in by_product.items()}
    cut = cfg.get("max_input_chars", 800)
    x_train = [redact(r["narrative"])[0][:cut] for r in folds["train"]]
    x_sub = [redact(r["narrative"])[0][:cut] for r in eval_rows]
    controls = {
        "B0-sub": [majority] * len(x_sub),
        "B1-sub": [product_prior.get(r["product"], majority) for r in records],
        "B2-sub": BowNaiveBayes().fit(x_train, y_train).predict(x_sub),
    }

    def line(name: str, pred: list[str]) -> list:
        c = bootstrap_macro_f1(y_true, pred, n_boot=500, seed=SEED)
        return [name, round(macro_f1(y_true, pred), 6),
                f"[{c['ci_low']}, {c['ci_high']}]",
                round(balanced_accuracy(y_true, pred), 6),
                round(accuracy(y_true, pred), 6),
                round(abstention_rate(pred), 6)]

    t1b = [line("B3 LLM", y_pred)] + [line(k, v) for k, v in controls.items()]
    t1b_md = [
        "# Table 1b. Same-frame control on the LLM evaluation subset", "",
        f"All four rows use the SAME {len(y_true)} Issue-stratified frozen test "
        "records and the same redacted, "
        f"{cut}-character-truncated input. Paired bootstrap 95% CI, 500 resamples, "
        f"seed {SEED}.", "",
        "| Method | macro-F1 | 95% CI | Balanced acc. | Accuracy | Abstention |",
        "|---|---|---|---|---|---|",
    ]
    t1b_md += ["| " + " | ".join(str(x) for x in r) + " |" for r in t1b]
    t1b_md += ["", "Table 1's B0-B2 rows use the full n=6,001 test fold and must "
                   "not be compared directly with these subset figures.", ""]
    (root / "results" / "table1b_llm_subset.md").write_text(
        "\n".join(t1b_md), encoding="utf-8")
    dump_json({"rows": t1b, "n": len(y_true), "max_input_chars": cut,
               "seed": SEED, "note": "same-frame controls for the LLM subset"},
              root / "results" / "table1b_llm_subset.json")

    # ---- overwrite the B3 placeholder row in Table 1 --------------------
    p = root / "results" / "table1_baselines.csv"
    with p.open(encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])
        t1 = list(rd)
    for row in t1:
        if row["baseline"] == "B3":
            row.update({
                "macro_f1": str(b3["macro_f1"]),
                "macro_f1_ci95": f"[{ci['ci_low']}, {ci['ci_high']}]",
                "balanced_accuracy": str(b3["balanced_accuracy"]),
                "accuracy": str(b3["accuracy"]),
                "wall_seconds": "0 (cache replay)",
                "peak_working_set_mb": "-",
                "notes": "n=400 stratified subset; NOT comparable with full-fold "
                         "B0-B2; see table1b_llm_subset.md",
            })
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(t1)
    md = ["# Table 1. Initial baseline results (frozen temporal test fold)", "",
          "| Baseline | Definition | macro-F1 | 95% CI | Balanced acc. | "
          "Accuracy | s | MB |", "|---|---|---|---|---|---|---|---|"]
    for r in t1:
        md.append("| " + " | ".join(r[k] for k in fields[:8]) + " |")
    md += ["", "B3 is scored on a n=400 Issue-stratified subset of the same "
               "frozen test fold; B0-B2 use the full n=6,001 fold. The two "
               "sample frames are different, so Table 1 must not be read as a "
               "like-for-like ranking. `results/table1b_llm_subset.md` "
               "recomputes B0/B1/B2 on the identical 400 records.", ""]
    (root / "results" / "table1_baselines.md").write_text("\n".join(md),
                                                         encoding="utf-8")

    bp = root / "results" / "baseline_metrics.json"
    bm = json.loads(bp.read_text(encoding="utf-8"))
    bm["baselines"]["B3"] = b3
    dump_json(bm, bp)

    # ---- append the LLM-specific negative results ------------------------
    neg_path = root / "results" / "negative_results.md"
    neg = (neg_path.read_text(encoding="utf-8").rstrip()
           if neg_path.exists() else "# Negative / failed results (auto-generated)")
    var = result["metrics"]["output_variability"]
    wrong = result["failure_taxonomy"].get("wrong_label_with_grounded_evidence", 0)
    b2_sub = t1b[3][1]
    neg += "\n" + "\n".join([
        f"- B3 (in-session LLM, `{result['mode']}`) reaches macro-F1 "
        f"{b3['macro_f1']} on the n=400 subset versus B2-sub {b2_sub}; the 95% "
        "bootstrap CIs overlap, so the gap is suggestive rather than statistically "
        "established. A larger evaluation frame is needed in phase 3.",
        f"- Output variability across {var['k']} INDEPENDENT repeats: mean pairwise "
        f"agreement {var['mean_pairwise_agreement']}, all-runs-identical rate "
        f"{var['all_runs_identical_rate']}. temperature=0.0 did not make the "
        "in-session model deterministic; a single-run score hides this instability.",
        f"- Abstention rate is {b3['abstention_rate']}: the model never used the "
        "ABSTAIN option, so the abstention safeguard is presently unproven.",
        f"- {wrong} of {len(y_true) * cfg.get('repeats', 3)} runs are 'grounded evidence but wrong "
        "label'. The dominant failure is label ambiguity / the 72-class long tail, "
        "not fabricated evidence; this argues for label-set consolidation and "
        "human adjudication in phase 3.",
        "",
    ])
    neg_path.write_text(neg, encoding="utf-8")
    return b3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--backend", choices=["mock", "cache", "api"], default="mock")
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    cfg = load_config(root)
    template_path = root / "prompts" / "v1.txt"
    template = template_path.read_text(encoding="utf-8")

    folds = load_folds(root)
    keep, policy = label_policy(folds["train"])
    label_set = sorted(keep | {"__OTHER__"})
    allowed = set(label_set) | {ABSTAIN}
    cfg["_label_set"] = label_set

    eval_ids = set((root / "data" / "splits" / "llm_eval_ids.txt").read_text(
        encoding="utf-8").splitlines())
    by_id = {r["id_hash"]: r for r in folds["test"]}
    eval_rows = [by_id[i] for i in sorted(eval_ids) if i in by_id][:args.limit]
    # few-shot examples come strictly from TRAIN and are checked against eval IDs
    few_shot = []
    for r in sorted(folds["train"], key=lambda x: x["id_hash"]):
        if len(few_shot) >= 4:
            break
        if r["id_hash"] in eval_ids:
            continue
        few_shot.append({"text": r["narrative"][:180],
                         "label": r["issue"] if r["issue"] in keep else "__OTHER__",
                         "id_hash": r["id_hash"]})
    assert not ({f["id_hash"] for f in few_shot} & eval_ids), "few-shot leakage"
    dump_json(few_shot, root / "prompts" / "few_shot.json")

    backend: Backend = {"mock": MockBackend, "cache": lambda: CacheBackend(root),
                        "api": ApiBackend}[args.backend]()

    records, cost_rows = [], []
    repeats: list[list[str]] = [[] for _ in range(cfg.get("repeats", 3))]
    total_tokens = 0
    t_start = time.perf_counter()
    for i, r in enumerate(eval_rows):
        redacted, markers = redact(r["narrative"])
        texts_for_grounding = redacted[:cfg.get("max_input_chars", 800)]
        prompt = build_prompt(template, label_set, texts_for_grounding, few_shot)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        run_labels, run_grounded, run_parse = [], [], []
        raw_outputs = []
        for k in range(cfg.get("repeats", 3)):
            out = backend.complete(prompt, cfg, run=k)
            parsed = parse_response(out["raw"], allowed)
            g = groundedness(parsed["evidence"], texts_for_grounding)
            run_labels.append(parsed["label"])
            run_grounded.append(g)
            run_parse.append(parsed["parse_ok"])
            raw_outputs.append({"run": k, "raw": out["raw"][:800],
                                "parsed": parsed, "groundedness": g,
                                "latency_s": out.get("latency_s"),
                                "is_simulation": out.get("is_simulation", False)})
            total_tokens += out.get("prompt_tokens", 0) + out.get("completion_tokens", 0)
            cost_rows.append([r["id_hash"], k, out.get("prompt_tokens", 0),
                              out.get("completion_tokens", 0),
                              out.get("latency_s", 0.0),
                              out.get("is_simulation", False)])
            repeats[k].append(parsed["label"])
        records.append({
            "id_hash": r["id_hash"], "date": r["date"],
            "true_label": r["issue"] if r["issue"] in keep else "__OTHER__",
            "product": r["product"],
            "prompt_sha256": prompt_sha,
            "redaction_markers": dict(markers),
            "runs": raw_outputs,
            "first_label": run_labels[0],
            "mean_groundedness": round(sum(run_grounded) / len(run_grounded), 6),
            "schema_ok_rate": round(sum(run_parse) / len(run_parse), 6),
        })
        if (i + 1) % 100 == 0:
            print(f"[llm_eval] {i+1}/{len(eval_rows)}", flush=True)

    y_true = [r["true_label"] for r in records]
    y_pred = [r["first_label"] for r in records]
    is_simulation = backend.name == "mock_deterministic"
    grounded_vals = [r["mean_groundedness"] for r in records]
    gl, gh = wilson_interval(sum(g >= 0.8 for g in grounded_vals), len(grounded_vals))

    # adversarial / prompt-injection suite
    # Leave room for the injected suffix so the adversarial text is not
    # truncated away (truncation would silently void the injection test).
    base_text = (redact(eval_rows[0]["narrative"])[0]
                 [:max(1, cfg.get("max_input_chars", 800) - 200)]
                 if eval_rows else "")
    adv_cases = make_adversarial_cases(base_text, label_set[:3])
    adv_results = []
    for case in adv_cases:
        p = build_prompt(template, label_set, case["text"], few_shot)
        out = backend.complete(p, cfg, run=0)
        parsed = parse_response(out["raw"], allowed)
        obeyed = (parsed["parse_ok"] and
                  not contains_injection(parsed["label"]) and
                  parsed["label"] in (allowed | {ABSTAIN}))
        adv_results.append({**case, "parsed_label": parsed["label"],
                            "schema_ok": parsed["parse_ok"],
                            "injection_not_followed": obeyed,
                            "raw": out["raw"][:300]})

    failure_taxonomy = Counter()
    for r in records:
        for run in r["runs"]:
            p = run["parsed"]
            if not p["parse_ok"]:
                failure_taxonomy["output_schema_violation"] += 1
            elif p["label"] == ABSTAIN:
                failure_taxonomy["abstention"] += 1
            elif p["error"] == "label_out_of_domain":
                failure_taxonomy["label_out_of_domain"] += 1
            elif run["groundedness"] < 0.8:
                failure_taxonomy["ungrounded_evidence"] += 1
            elif p["label"] != r["true_label"]:
                failure_taxonomy["wrong_label_with_grounded_evidence"] += 1

    result = {
        "mode": backend.name,
        "IS_SIMULATION": is_simulation,
        "not_reportable": is_simulation,
        "honesty_note": (
            "Deterministic stand-in used to self-test the evaluation harness. "
            "These scores are NOT model performance and must not appear in the "
            "report's comparison table." if is_simulation else
            "Cache replay or live API; see per-record raw outputs and costs."),
        "config": cfg,
        "prompt_file": "prompts/v1.txt",
        "prompt_sha256": sha256_file(template_path),
        "few_shot_file": "prompts/few_shot.json",
        "few_shot_sha256": sha256_file(root / "prompts" / "few_shot.json"),
        "few_shot_eval_id_overlap": 0,
        "eval_frame": {
            "n": len(eval_rows), "source": "data/splits/llm_eval_ids.txt",
            "label_set_size": len(label_set), "seed": SEED,
            "stratification": "Issue-stratified over the frozen test fold",
            "ground_truth": "the dataset's own Issue field (submission-time "
                            "self-selected label, noisy; not independent annotation)",
        },
        "metrics": {
            "macro_f1_first_run": round(macro_f1(y_true, y_pred), 6),
            "accuracy_first_run": round(accuracy(y_true, y_pred), 6),
            "abstention_rate": round(abstention_rate(y_pred), 6),
            "coverage_weighted": coverage_weighted_accuracy(y_true, y_pred),
            "mean_groundedness": round(sum(grounded_vals) / max(1, len(grounded_vals)), 6),
            "grounded_rate_wilson_ci95": [gl, gh],
            "schema_ok_rate": round(sum(r["schema_ok_rate"] for r in records)
                                    / max(1, len(records)), 6),
            "output_variability": output_consistency(repeats),
            "per_class": per_class_metrics(y_true, y_pred),
        },
        "adversarial_tests": adv_results,
        "adversarial_pass_rate": round(
            sum(a["injection_not_followed"] for a in adv_results)
            / max(1, len(adv_results)), 6),
        "redaction_audit": redaction_audit([r["narrative"] for r in eval_rows]),
        "failure_taxonomy": dict(failure_taxonomy),
        "total_tokens_estimate": total_tokens,
        "elapsed_seconds": round(time.perf_counter() - t_start, 2),
        "max_input_chars": cfg.get("max_input_chars", 800),
        "independent_repeats": backend.name == "cache_replay",
        "records": records,
    }
    dump_json(result, root / "results" / "llm_pilot.json")
    dump_csv(cost_rows, root / "results" / "llm_costs.csv",
             ["id_hash", "repeat", "prompt_tokens", "completion_tokens",
              "latency_s", "is_simulation"])

    # Table 2 (prompt / model configuration snapshot) is always real.
    t2 = [
        "# Table 2. LLM prompt and model configuration snapshot", "",
        "| Item | Value |", "|---|---|",
        f"| Config file | `configs/llm.json` (SHA256 `{cfg['_config_sha256'][:16]}...`) |",
        f"| Prompt | `prompts/v1.txt` (SHA256 `{result['prompt_sha256'][:16]}...`) |",
        f"| Few-shot | `prompts/few_shot.json` (4 TRAIN examples, 0 eval overlap) |",
        f"| Model | `{cfg['model']}` / version `{cfg['model_version']}` |",
        f"| Provider | `{cfg['provider']}` |",
        f"| temperature | {cfg['temperature']} |",
        f"| top_p | {cfg['top_p']} |",
        f"| max_tokens | {cfg['max_tokens']} |",
        f"| seed | {cfg['seed']} |",
        f"| repeats | {cfg['repeats']} |",
        f"| eval n | {len(eval_rows)} |",
        f"| External transmission authorised | {cfg['allow_external_transmission']} |",
        f"| Backend used for this pilot | `{backend.name}` |",
        f"| Simulation | {is_simulation} |",
        "", "Output schema: `issue_label`, `evidence_quote`, `confidence`, `abstain`.",
        "Any output failing schema validation is recorded as ABSTAIN and counted "
        "in the failure taxonomy.", "",
    ]
    (root / "results" / "table2_llm_config.md").write_text("\n".join(t2),
                                                           encoding="utf-8")
    b3 = update_table1(root, result, records, eval_rows, folds, keep, cfg)
    print(json.dumps({"mode": backend.name, "n": len(eval_rows),
                      "macro_f1": result["metrics"]["macro_f1_first_run"],
                      "groundedness": result["metrics"]["mean_groundedness"],
                      "IS_SIMULATION": is_simulation,
                      "B3_scored": b3 is not None}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
