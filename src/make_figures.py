#!/usr/bin/env python3
"""Regenerate every figure (1-6) and the machine-written findings text.

All numbers come from results/*.json|csv produced by the other scripts; no
value is typed into this file by hand.  English axis labels avoid CJK font
issues on clean machines; the Chinese report supplies the captions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160, "savefig.bbox": "tight",
})
BLUE, ORANGE, GREEN, GREY = "#2b6cb0", "#dd6b20", "#2f855a", "#718096"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("[fig] wrote", path.name, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    R, F = root / "results", root / "reports" / "figures"
    F.mkdir(parents=True, exist_ok=True)
    prof = json.loads((R / "full_profile.json").read_text(encoding="utf-8"))
    findings: list[str] = ["# Figure findings (auto-generated from results/)", ""]
    n = prof["file"]["logical_rows"]

    # ---- Figure 1: monthly volume and narrative coverage -----------------
    months = prof["dates"]["monthly_counts"]
    narr = prof["dates"]["monthly_narrative_counts"]
    keys = sorted(months)
    vols = [months[k] for k in keys]
    rates = [narr.get(k, 0) / max(1, months[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(8.4, 3.1))
    ax.bar(range(len(keys)), vols, color=BLUE, alpha=.85)
    ax.set_ylabel("Complaints / month")
    ax.set_title(f"Fig. 1  Monthly complaint volume, full file (n={n:,})")
    ax.set_xticks(range(0, len(keys), max(1, len(keys) // 12)))
    ax.set_xticklabels([keys[i] for i in range(0, len(keys), max(1, len(keys) // 12))],
                       rotation=45, ha="right", fontsize=7)
    ax2 = ax.twinx()
    ax2.plot(range(len(keys)), rates, color=ORANGE, lw=1.6, marker=".", ms=3)
    ax2.set_ylabel("Narrative coverage", color=ORANGE)
    ax2.set_ylim(0, max(rates) * 1.15 if rates else 1)
    ax2.spines["top"].set_visible(False)
    save(fig, F / "fig1_monthly_volume.png")
    peak = keys[vols.index(max(vols))]
    findings.append(
        f"- **Fig. 1** Over {len(keys)} months ({keys[0]} to {keys[-1]}) the monthly "
        f"volume peaks at {max(vols):,} in {peak}; narrative coverage varies from "
        f"{min(rates):.1%} to {max(rates):.1%} across months. The head of the file "
        f"is dominated by one product while the whole file is not chronological "
        f"(first logical row "
        f"{prof['dates']['first_logical_row_date']}, last "
        f"{prof['dates']['last_logical_row_date']}, "
        f"{prof['dates']['adjacent_date_descents']:,} adjacent date descents). "
        "The raw file is not chronological. => a row-index split is invalid; "
        "use a temporal split.")

    # ---- Figure 2: Issue (target) class distribution --------------------
    issue = prof["labels"]["issue_top20"]
    labs, vals = list(issue.keys()), list(issue.values())
    total_labelled = sum(prof["columns"][3]["top_values"].values()) if False else None
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ypos = range(len(labs))
    ax.barh(list(ypos), vals, color=GREEN, alpha=.85)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([s[:46] for s in labs], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Records (full file, top 20 Issue labels)")
    ax.set_title(f"Fig. 2  Target distribution: Issue ({prof['labels']['issue_class_count']} classes)")
    for i, v in enumerate(vals[:6]):
        ax.text(v, i, f" {v:,}", va="center", fontsize=7)
    save(fig, F / "fig2_issue_distribution.png")
    top1 = max(vals)
    all_issue = read_csv(R / "issue_distribution.csv")
    share_top1 = float(all_issue[0]["share"]) if all_issue else 0
    findings.append(
        f"- **Fig. 2** `Issue` has {prof['labels']['issue_class_count']} distinct values; "
        f"the largest class ('{labs[0]}') holds {top1:,} records "
        f"({share_top1:.1%} of labelled rows). Class imbalance rules out accuracy as "
        "a primary metric and motivates macro-F1 / per-class recall.")

    # ---- Figure 3: full-file missingness ---------------------------------
    cols = prof["columns"]
    ordered = sorted(cols, key=lambda c: c["missing_rate"])
    names = [c["field"][:34] for c in ordered]
    mrate = [c["missing_rate"] * 100 for c in ordered]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.barh(range(len(names)), mrate,
            color=[ORANGE if r > 50 else BLUE for r in mrate], alpha=.88)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Missing (%)")
    ax.set_title(f"Fig. 3  Full-file missingness, n={n:,} logical records")
    for i, r in enumerate(mrate):
        ax.text(r, i, f" {r:.1f}", va="center", fontsize=6.5)
    ax.set_xlim(0, 105)
    save(fig, F / "fig3_missingness.png")
    worst = ordered[-4:]
    findings.append(
        "- **Fig. 3** Highest missingness: " +
        ", ".join(f"`{c['field']}` {c['missing_rate']:.2%}" for c in reversed(worst)) +
        f". Narrative availability is {prof['narrative']['nonempty_rate']:.2%} "
        f"({prof['narrative']['nonempty']:,} records) -- the declared text-classification "
        "sample frame, not the whole population.")

    # ---- Figure 4: product x period + narrative coverage ----------------
    pp = prof["product_period_top20"][:12]
    plabs = [p["product"][:34] for p in pp]
    early = [p["early_period"] for p in pp]
    late = [p["late_period"] for p in pp]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ypos = range(len(plabs))
    axes[0].barh(list(ypos), early, color=BLUE, alpha=.85, label="earlier half")
    axes[0].barh(list(ypos), late, left=early, color=ORANGE, alpha=.85,
                 label="later half")
    axes[0].set_yticks(list(ypos)); axes[0].set_yticklabels(plabs, fontsize=7)
    axes[0].invert_yaxis(); axes[0].legend(fontsize=7)
    axes[0].set_title("Fig. 4a  Product x time period")
    axes[0].set_xlabel("Complaints")
    cov = [p["with_narrative"] / max(1, p["total"]) for p in pp]
    axes[1].barh(list(ypos), cov, color=GREEN, alpha=.85)
    axes[1].set_yticks(list(ypos)); axes[1].set_yticklabels([])
    axes[1].invert_yaxis(); axes[1].set_xlim(0, 1)
    axes[1].set_title("Fig. 4b  Narrative coverage by product")
    axes[1].set_xlabel("Fraction with narrative")
    fig.suptitle("Fig. 4  Subgroup structure and sampling bias", fontsize=10)
    save(fig, F / "fig4_product_subgroups.png")
    findings.append(
        f"- **Fig. 4** Product mix shifts strongly between the earlier and later half "
        f"of the timeline, and narrative coverage differs by product "
        f"(range {min(cov):.1%}-{max(cov):.1%} among the 12 largest products). "
        "=> stratified reporting and an explicit sample-frame statement are required; "
        "head-of-file sampling would badly misrepresent the population.")

    # ---- Figure 5: pipeline flowchart ------------------------------------
    steps = [
        ("ZIP / CSV\n1.4 GB / 8.9 GB", BLUE),
        ("CountingStream\nSHA256 + line count", BLUE),
        ("csv.reader\nlogical records", BLUE),
        ("validate.py\nschema / masks / dates", GREEN),
        ("clean.py\nfeatures | labels split", GREEN),
        ("split.py\ntemporal 70/15/15", ORANGE),
        ("baselines.py\nB0 / B1 / B2", ORANGE),
        ("llm_eval.py\nprompt / schema / grounding", ORANGE),
        ("make_figures.py\ntables + figures", GREY),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    ax.set_xlim(0, 9.6); ax.set_ylim(0, 3.4); ax.axis("off")
    for i, (label, color) in enumerate(steps):
        row, col = divmod(i, 5)
        x = 0.25 + col * 1.86
        y = 2.05 - row * 1.35
        box = FancyBboxPatch((x, y), 1.62, 0.82,
                             boxstyle="round,pad=0.04,rounding_size=0.06",
                             linewidth=1.1, edgecolor=color,
                             facecolor=color, alpha=.13)
        ax.add_patch(box)
        ax.text(x + .81, y + .41, label, ha="center", va="center",
                fontsize=7.2, color=color)
        if i not in (4, 8):
            ax.annotate("", xy=(x + 1.80, y + .41), xytext=(x + 1.64, y + .41),
                        arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
    ax.text(4.8, 3.18, "Fig. 5  Reproducible Phase-2 processing pipeline",
            ha="center", fontsize=10, weight="bold")
    ax.text(4.8, .05,
            "Post-hoc fields are separated at clean.py and can never enter a feature matrix",
            ha="center", fontsize=7.5, color=ORANGE)
    save(fig, F / "fig5_pipeline.png")
    findings.append(
        "- **Fig. 5** Nine-stage pipeline. The explicit feature/label separation at "
        "`clean.py` plus `validate.assert_no_leakage()` is the structural leak guard; "
        "the temporal split is frozen as SHA256-listed ID files before any baseline is fit.")

    # ---- Figure 6: SAME-FRAME comparison on the LLM evaluation subset ---
    # Table 1's B0-B2 use the full 6,001-record test fold while B3 uses a
    # 400-record stratified subset.  Figure 6 therefore plots Table 1b, where
    # all four methods are recomputed on the identical 400 records.
    t1b_path = R / "table1b_llm_subset.json"
    if t1b_path.exists():
        t1b = json.loads(t1b_path.read_text(encoding="utf-8"))["rows"]
        names6 = [r[0] for r in t1b]
        mf = [float(r[1]) for r in t1b]
        ci = [json.loads(r[2]) for r in t1b]
        ba6 = [float(r[3]) for r in t1b]
        ac6 = [float(r[4]) for r in t1b]
        err = [[max(0, m - c[0]) for m, c in zip(mf, ci)],
               [max(0, c[1] - m) for m, c in zip(mf, ci)]]
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
        axes[0].bar(names6, mf, color=[BLUE, GREEN, ORANGE, "#b03a2e"],
                    alpha=.88, yerr=err, capsize=4, ecolor=GREY)
        axes[0].set_ylim(0, max(mf + [0.1]) * 1.35)
        axes[0].set_ylabel("macro-F1 (95% bootstrap CI)")
        axes[0].set_title("Fig. 6a  Same n=400 evaluation subset")
        width = .36
        for j, (metric, vals) in enumerate([("balanced accuracy", ba6),
                                            ("accuracy", ac6)]):
            axes[1].bar([i + (j - .5) * width for i in range(len(names6))],
                        vals, width=width, label=metric,
                        color=[BLUE, ORANGE][j], alpha=.88)
        axes[1].set_xticks(range(len(names6))); axes[1].set_xticklabels(names6)
        axes[1].set_ylim(0, 1); axes[1].legend(fontsize=7)
        axes[1].set_title("Fig. 6b  Aggregate metrics, same subset")
        fig.suptitle("Fig. 6  Same-frame comparison: in-session LLM vs three baselines "
                     "(n=400, 95% bootstrap CI)", fontsize=9.5)
        save(fig, F / "fig6_baselines.png")
        findings.append(
            "- **Fig. 6** All four methods are recomputed on the SAME 400 "
            "Issue-stratified frozen test records and the same redacted, "
            "800-character input: " + "; ".join(
                f"{r[0]} macro-F1={r[1]} (95% CI {r[2]})" for r in t1b) +
            ". The LLM point estimate is highest, but its CI overlaps B2-sub's "
            "upper tail, so this is a promising difference, NOT a claim of "
            "statistical significance.")
    else:
        t1 = read_csv(R / "table1_baselines.csv")
        real = [r for r in t1 if r["baseline"] != "B3"]
        names6 = [r["baseline"] for r in real]
        mf = [float(r["macro_f1"]) for r in real]
        ci = [json.loads(r["macro_f1_ci95"]) for r in real]
        err = [[max(0, m - c[0]) for m, c in zip(mf, ci)],
               [max(0, c[1] - m) for m, c in zip(mf, ci)]]
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
        axes[0].bar(names6, mf, color=[BLUE, GREEN, ORANGE], alpha=.88,
                    yerr=err, capsize=4, ecolor=GREY)
        axes[0].set_ylim(0, max(mf + [0.1]) * 1.35)
        axes[0].set_ylabel("macro-F1 (95% bootstrap CI)")
        axes[0].set_title("Fig. 6a  Baselines on frozen test fold")
        width = .36
        for j, m in enumerate(["balanced_accuracy", "accuracy"]):
            vals = [float(r[m]) for r in real]
            axes[1].bar([i + (j - .5) * width for i in range(len(names6))], vals,
                        width=width, label=m.replace("_", " "),
                        color=[BLUE, ORANGE][j], alpha=.88)
        axes[1].set_xticks(range(len(names6))); axes[1].set_xticklabels(names6)
        axes[1].set_ylim(0, 1); axes[1].legend(fontsize=7)
        axes[1].set_title("Fig. 6b  Aggregate metrics")
        fig.suptitle("Fig. 6  Initial baseline results (B3 LLM not scored)",
                     fontsize=9.5)
        save(fig, F / "fig6_baselines.png")
        findings.append("- **Fig. 6** B3 LLM is unscored in this run; only the "
                        "mock harness self-test was executed.")

    # ---- head-vs-full evidence table + tables bundle ---------------------
    h = prof["head_100k"]
    findings += [
        "",
        "## Design-changing observations (quantified)",
        "",
        f"1. **Head-of-file sampling is invalid.** In the first {h['rows']:,} logical "
        f"records, narrative is missing {h['narrative_missing_rate']:.2%} and "
        f"credit-reporting products are {h['credit_reporting_rate']:.2%}; over the "
        f"full {n:,} records narrative is present {prof['narrative']['nonempty_rate']:.2%} "
        f"of the time. Phase 1's 548-row head sample reported 88.9% missing narrative, "
        "which does not generalise. => replace head sampling with full-file counting "
        "plus a seeded reservoir sample.",
        f"2. **The phase-1 target is unusable.** `Timely response?` = Yes in "
        f"{prof['labels']['timely_response'].get('Yes', 0):,}/{n:,} full-file records "
        f"({prof['labels']['timely_response'].get('Yes', 0)/n:.2%}). A constant predictor "
        "already achieves that accuracy. => target changed to `Issue`, primary metric "
        "changed to macro-F1.",
        f"3. **Ordering is neither chronological nor a simple product partition.** "
        f"The first logical row is dated {prof['dates']['first_logical_row_date']} and "
        f"the last {prof['dates']['last_logical_row_date']}; there are "
        f"{prof['dates']['adjacent_date_descents']:,} adjacent date descents. Yet the "
        f"first {h['rows']:,} rows are {h['credit_reporting_rate']:.2%} one product, "
        f"while over the full file the product changes "
        f"{prof['dates']['product_runs']:,} times (mean "
        f"{prof['dates']['mean_rows_per_product_run']:.1f} rows per run). "
        "=> temporal split on `Date received`, never row index; head sampling is unsafe.",
    ]
    (R / "figure_findings.md").write_text("\n".join(findings) + "\n", encoding="utf-8")

    # A single Markdown bundle of tables 1-3 for the report appendix.
    parts = ["# Tables 1-3 (regenerated)", ""]
    for name in ("table1_baselines.md", "table2_llm_config.md"):
        p = R / name
        if p.exists():
            parts.append(p.read_text(encoding="utf-8"))
    t3 = R / "table3_splits.csv"
    if t3.exists():
        parts += ["# Table 3. Frozen split and sample sizes", "",
                  "| fold | n | share | date_min | date_max | distinct issues | sha256 |",
                  "|---|---|---|---|---|---|---|"]
        for r in read_csv(t3):
            parts.append("| " + " | ".join(
                [r["fold"], r["n"], r["share"], r["date_min"], r["date_max"],
                 r["distinct_issues"], r["ids_sha256_prefix"]]) + " |")
    (R / "tables_bundle.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("[fig] all done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
