#!/usr/bin/env python3
"""End-to-end orchestrator: regenerate every Phase-2 artifact from source.

    python src/run_all.py --source /path/to/complaints.csv.zip --root .
    python src/run_all.py --root . --skip-profile     # reuse committed results
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], root: Path, name: str) -> None:
    print(f"\n===== [{name}] {' '.join(cmd)} =====", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=root)
    if proc.returncode != 0:
        raise SystemExit(f"[{name}] FAILED with exit code {proc.returncode}")
    print(f"===== [{name}] done in {time.perf_counter() - t0:.1f}s =====\n", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--source", default="")
    ap.add_argument("--skip-profile", action="store_true",
                    help="reuse results/full_profile.json (offline reproduction)")
    ap.add_argument("--backend", default="mock", choices=["mock", "cache", "api"])
    args = ap.parse_args()
    root = Path(args.root).resolve()
    py = sys.executable

    run([py, "src/make_sample.py"], root, "sample")
    if not args.skip_profile:
        if not args.source:
            raise SystemExit("--source is required unless --skip-profile is used")
        run([py, "src/profile_full.py", "--source", args.source,
             "--out-dir", str(root), "--sample-size", "40000"], root, "profile")
    run([py, "src/split.py", "--root", str(root)], root, "split")
    run([py, "src/baselines.py", "--root", str(root)], root, "baselines")
    run([py, "src/llm_eval.py", "--root", str(root), "--backend", args.backend],
        root, "llm_eval")
    run([py, "src/make_figures.py", "--root", str(root)], root, "figures")
    run([py, "src/make_docs.py", "--root", str(root)], root, "docs")
    run([py, "-m", "pytest", "-q"], root, "tests")
    print("ALL PHASE-2 ARTIFACTS REGENERATED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
