# Reproduction commands (Phase 2)

```bash
python -m pip install -r requirements.txt
python src/profile_full.py --source <complaints.csv.zip> --out-dir . --sample-size 40000
python src/split.py --root .
python src/baselines.py --root .
python src/llm_eval.py --root . --backend mock
python src/make_figures.py --root .
python src/make_docs.py --root .
python -m pytest -q
```

Replace `--backend mock` with `--backend cache` or `--backend api` when an authorised model endpoint is available. Mock numbers are never reportable model results.

Full-file profile: 16,545,376 logical rows in 580.84 s, peak working set 161 MiB.
