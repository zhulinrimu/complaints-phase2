# Phase 2 — CFPB Consumer Complaint Database: data & experiment design

Reproducible companion repository for `gti_phase2_report.pdf`.

- Submission date: 2026-11-07
- Repository: <https://github.com/zhulinrimu/complaints-phase2>, tag `phase-2-submission`
- The submission PDF, its editable DOCX and the status checklist are submitted
  separately and are intentionally **not** part of this repository: only code,
  configuration, frozen split IDs and regenerated evidence are versioned here.
- Phase 1: <https://github.com/zhulinrimu/complaints-phase1> (tag `phase-1-submission`,
  commit `38a6367d78c1541c813c49b2b3d09b1e1ba4b821`).

## What changed since Phase 1

| Phase 1 | Phase 2 |
|---|---|
| 548-row head sample | single-pass profile of the complete 8.9 GB file |
| `TextIOWrapper` without `newline=''`, so quoted newlines split records | `CountingStream` + `newline=''`: logical records, physical lines, CSV SHA256 in one pass |
| `seen_ids` set (O(rows) RAM) | exact duplicate counting via 64 disk buckets of 128-bit digests |
| ZIP privacy masks counted as format errors | `XXXXX` / `NNNXX` recorded as `masked`, not errors |
| no target, no split, no baseline | target `Issue`, frozen temporal split, B0/B1/B2 baselines with bootstrap CIs |
| LLM only used to help write code | LLM is an evaluated method: versioned prompt, JSON schema, groundedness, abstention, variability, prompt-injection tests |

## Setup

Python >= 3.11. The complete raw-data pipeline uses **only the standard library**.
Figures and the test runner need three extra packages:

```bash
python -m pip install -r requirements.txt
```

The raw file is **not** redistributed. Obtain `complaints.csv.zip` from the CFPB
Consumer Complaint Database and record the download date/URL/snapshot in
`data/dataset_card.md`. A small synthetic sample (`data/sample_complaints.csv`)
is included for tests and for a runnable demo without the 8.9 GB file.

## One-command reproduction (full data, requires the zip)

```bash
python src/run_all.py --source /path/to/complaints.csv.zip --root .
```

This executes, in order: `profile_full.py` -> `split.py` -> `baselines.py` ->
`llm_eval.py` -> `make_figures.py` -> `make_docs.py` -> `pytest -q`.
Everything in `results/` and `reports/figures/` is regenerated from code.

## Runnable demo without the raw file

```bash
python -m pytest -q
python src/make_figures.py --root .        # uses committed results/*.json
python src/llm_eval.py --root . --backend mock --limit 20
```

`--backend mock` is a **deterministic harness self-test**, not a language model.
Its output is flagged `IS_SIMULATION: true` and `not_reportable: true`; those
numbers must never be quoted as model performance.

## Layout

```
src/
  common.py         streaming IO, SHA256, physical/logical line counting, schema
  profile_full.py   single-pass full-file statistics + seeded reservoir sample
  validate.py       row-level validation rules and a hard leakage guard
  clean.py          normalisation; features and post-hoc labels kept separate
  split.py          frozen temporal 70/15/15 split + frozen LLM eval IDs
  metrics.py        macro-F1, bootstrap CI, groundedness, variability, Wilson CI
  baselines.py      B0 majority / B1 prior lookup / B2 bag-of-words Naive Bayes
  redact.py         PII redaction + prompt-injection test cases
  llm_eval.py       LLM evaluation harness (mock / cache / live API backends)
  make_figures.py   figures 1-6 and tables 1-3
  make_docs.py      auto-generated data dictionary, dataset card, environment lock
  run_all.py        end-to-end orchestrator
tests/               >= 8 meaningful tests (run with pytest -q)
configs/             phase2.json, llm.json (seeds, thresholds, model config)
prompts/             v1.txt, few_shot.json, adversarial cases
data/                small sample, derived sample, frozen split ID lists
results/             JSON/CSV evidence, pilot outputs, cost log
reports/figures/     PNG figures
```

## Ethical / authorisation gate

Complaint narratives are public but de-identified text that can still contain
names, account numbers and addresses. `src/redact.py` replaces likely PII before
any external call. **No live API call is made by default**:
`configs/llm.json` has `allow_external_transmission: false`. Set it to true only
after the team authorises third-party transmission, then pin the exact model
version and set `LLM_API_KEY`.

## AI-use disclosure

See appendix E of the report. In short: an AI assistant helped write code and
report drafts; all full-data runs, tests, numbers and interpretations were
executed and verified by the team. The LLM evaluated inside the system is a
separate, explicitly disclosed component. No simulated result is presented as
a live model result.

Team contribution (provided 2026-11-07): 朱建坤 designed the report draft with
LLM tools and created the GitHub repository; 郑智 reviewed the draft and
verified the reference citations. See section 12 and appendix E of the report.
