# Reproduction commands (Phase 2)

```bash
python -m pip install -r requirements.txt
python src/profile_full.py --source <complaints.csv.zip> --out-dir . --sample-size 40000
python src/split.py --root .
python src/baselines.py --root .
python src/llm_eval.py --root . --backend cache
python src/make_figures.py --root .
python src/make_docs.py --root .
python -m pytest -q
```

`--backend cache` is the DEFAULT reproducible path: `results/llm_cache.jsonl` holds 1,203 verbatim in-session LLM responses (400 records x 3 independent repeats + 3 adversarial cases), keyed by prompt SHA256. Replaying it regenerates every reported B3 number without any network call. Course guide p.25 explicitly permits cached evaluation artefacts. `--backend mock` is a deterministic harness self-test and its numbers are never reportable; `--backend api` needs `LLM_API_KEY` plus a pinned model version and is reserved for phase 3 cross-model comparison.

Full-file profile: 16,545,376 logical rows in 580.84 s, peak working set 161 MiB.
