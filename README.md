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
| LLM only used to help write code | LLM is an **evaluated method**: 1,203 real in-session inferences, cached verbatim and replayed, scored against three baselines on an identical frame |

## Headline results

Full-file profile: **16,545,376** logical records, **27,656,645** physical lines,
76.94% missing narratives, `Timely response?` = Yes in 99.38% of rows (hence
rejected as a target), 180 `Issue` classes. Two independent full runs agree on
18 key statistics.

| Method | Frame | macro-F1 | 95% CI |
|---|---|---|---|
| B0 majority | full test fold n=6,001 | 0.0101 | [0.0099, 0.0105] |
| B1 Product→Issue prior | full test fold n=6,001 | 0.0938 | [0.0889, 0.1000] |
| B2 bag-of-words NB | full test fold n=6,001 | 0.1556 | [0.1409, 0.1727] |
| **B3 LLM** | **stratified subset n=400** | **0.1706** | **[0.1284, 0.1921]** |

Table 1 and this B3 row use **different sample frames and are NOT comparable**.
`results/table1b_llm_subset.md` recomputes all four methods on the identical
400 records and identical redacted, 800-character input: B3 0.1706, B2 0.1138,
B1 0.0662, B0 0.0007. The LLM point estimate is highest but its CI overlaps
B2's, so no statistical significance is claimed. Figure 6 plots this same-frame
comparison.

Other measured LLM properties: groundedness 1.00 (Wilson CI [0.9905, 1.0000]),
JSON-schema compliance 1.00, prompt-injection tests 3/3 not followed,
abstention rate 0.00, and **real** output variability across three independent
repeats (mean pairwise agreement 0.719, all-identical rate 0.615). 962 of 1,200
runs were "grounded evidence but wrong label", pointing to label noise and the
72-class long tail rather than hallucination.

## Setup

Python >= 3.11. The complete raw-data pipeline uses **only the standard library**.
Figures and the test runner need three extra packages:

```bash
python -m pip install -r requirements.txt
```

The raw file is **not** redistributed. Obtain `complaints.csv.zip` from the CFPB
Consumer Complaint Database (<https://www.consumerfinance.gov/data-research/consumer-complaints/>).
Dataset identity is pinned by the SHA256 of the uncompressed CSV recorded in
`data/dataset_card.md` rather than by a download date. A small synthetic sample
(`data/sample_complaints.csv`) is included for tests and for a runnable demo
without the 8.9 GB file.

## One-command reproduction (full data, requires the zip)

```bash
python src/run_all.py --source /path/to/complaints.csv.zip --root . --backend cache
```

This executes, in order: `profile_full.py` -> `split.py` -> `baselines.py` ->
`llm_eval.py` -> `make_figures.py` -> `make_docs.py` -> `pytest -q`.
Everything in `results/` and `reports/figures/` is regenerated from code.
Use `--skip-profile` to reuse the committed full-file statistics.

**`--backend cache` is the default reproducible path.** `results/llm_cache.jsonl`
stores 1,203 verbatim in-session LLM responses (400 records x 3 independent
repeats + 3 adversarial cases), keyed by the SHA256 of the exact prompt.
`results/llm_cache_manifest.json` records the model identity, response-file
hashes, redaction policy and validation counts. Replaying the cache regenerates
every reported B3 number with no network call and no API key. The course guide
(p.25) explicitly permits cached evaluation artifacts when an external API is
unavailable.

`--backend mock` is a **deterministic harness self-test**, not a language model;
its output is flagged `IS_SIMULATION: true` and `not_reportable: true` and must
never be quoted as model performance. `--backend api` needs `LLM_API_KEY`,
`allow_external_transmission: true` and a pinned model version; it is reserved
for the phase-3 cross-model comparison.

## Runnable demo without the raw file

```bash
python -m pytest -q
python src/make_figures.py --root .        # uses committed results/*.json
python src/llm_eval.py --root . --backend cache
```

### Reproducibility tiers

The 40,000-row derived narrative sample (`data/derived/narrative_sample.csv.gz`)
contains complaint narratives and is intentionally **not** versioned
(`.gitignore`); it is written by `src/profile_full.py --source <complaints.csv.zip>`.
The commands therefore fall into three tiers:

| Tier | Commands | Needs |
|---|---|---|
| Clean clone, no raw data | `pytest -q`, `src/make_figures.py`, `src/make_docs.py`, cache/manifest hash checks | `requirements.txt` only |
| Local derived sample | `src/split.py`, `src/baselines.py`, `src/llm_eval.py --backend cache`, `src/run_all.py --skip-profile --backend cache` | the derived `.csv.gz` above |
| Full pipeline | `src/run_all.py --source <complaints.csv.zip> --root . --backend cache` | the 1.4 GB CFPB archive |

In a clean clone the split/baseline test checks the committed `label_policy`
against the frozen split manifest instead of reopening plaintext narratives;
`pytest -q` still reports 39 passed. The B3 scores are already committed in
`results/llm_pilot.json` and `results/table1b_llm_subset.md`; replaying the
response cache with `llm_eval.py` needs the derived sample because prompts are
rebuilt from the narratives. Figures and documents are regenerated from the
committed `results/*.json` and remain byte-for-byte reproducible without it.

## Layout

```
src/
  common.py             streaming IO, SHA256, physical/logical line counting, schema
  profile_full.py       single-pass full-file statistics + seeded reservoir sample
  validate.py           row-level validation rules and a hard leakage guard
  clean.py              normalisation; features and post-hoc labels kept separate
  split.py              frozen temporal 70/15/15 split + frozen LLM eval IDs
  metrics.py            macro-F1, bootstrap CI, groundedness, variability, Wilson CI
  baselines.py          B0 majority / B1 prior lookup / B2 bag-of-words Naive Bayes
  redact.py             PII redaction + prompt-injection test cases
  llm_eval.py           LLM evaluation harness (mock / cache / live API backends)
  build_llm_cache.py    prepare response batches, validate + assemble the cache
  make_figures.py       figures 1-6 and tables 1, 1b, 2, 3
  make_docs.py          auto-generated data dictionary, dataset card, environment lock
  run_all.py            end-to-end orchestrator
tests/                   39 tests (run with pytest -q)
configs/                 phase2.json, llm.json (seeds, thresholds, model config)
prompts/                 v1.txt, few_shot.json
data/                    small sample, derived sample, frozen split ID lists
results/                 JSON/CSV evidence, LLM response cache + manifest, cost log
reports/figures/         PNG figures
```

## Ethical / authorisation gate

Complaint narratives are public but de-identified text that can still contain
names, account numbers and addresses. `src/redact.py` replaces likely PII before
any model call. The team authorised third-party transmission on 2026-11-07, but
**no external transmission was needed or performed**: all inference happened
inside the WorkBuddy session. The cache stores only labels and short verbatim
evidence quotes; full complaint text and the annotation batches are kept out of
this repository. `configs/llm.json` keeps `allow_external_transmission: false`,
so `api` calls are still refused until a key and an exact pinned model version
are supplied. The platform does not expose a pinned snapshot id; reproducibility
therefore rests on the response cache plus prompt/config SHA256 hashes, and this
limitation is disclosed in appendix E of the report.

## AI-use disclosure

See appendix E of the report. In short: an AI assistant helped write code and
report drafts; all full-data runs, tests, numbers and interpretations were
executed and verified by the team. The LLM evaluated inside the system is a
separate, explicitly disclosed component: the in-session model produced the
1,203 cached responses that B3 is scored from. No simulated result is presented
as a model result, and no independent human annotation or Cohen's kappa was used
in phase 2 (deferred to phase 3).

Team contribution (provided 2026-11-07): 朱建坤 designed the report draft with
LLM tools and created the GitHub repository; 郑智 reviewed the draft and
verified the reference citations. See section 12 and appendix E of the report.
