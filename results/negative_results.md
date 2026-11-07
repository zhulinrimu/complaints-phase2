# Negative / failed results (auto-generated)

- B2 macro-F1 is 0.155597: a bag-of-words model cannot separate the long tail of Issue labels. Decision: phase 3 must report per-class recall, not only an aggregate number.
- Constant prediction reaches accuracy 0.346609 on this label distribution while its macro-F1 is 0.010094. This is the empirical reason accuracy is not a primary metric.
- B3 (in-session LLM, `cache_replay`) reaches macro-F1 0.170558 on the n=400 subset versus B2-sub 0.113823; the 95% bootstrap CIs overlap, so the gap is suggestive rather than statistically established. A larger evaluation frame is needed in phase 3.
- Output variability across 3 INDEPENDENT repeats: mean pairwise agreement 0.719167, all-runs-identical rate 0.615. temperature=0.0 did not make the in-session model deterministic; a single-run score hides this instability.
- Abstention rate is 0.0: the model never used the ABSTAIN option, so the abstention safeguard is presently unproven.
- 962 of 1200 runs are 'grounded evidence but wrong label'. The dominant failure is label ambiguity / the 72-class long tail, not fabricated evidence; this argues for label-set consolidation and human adjudication in phase 3.
