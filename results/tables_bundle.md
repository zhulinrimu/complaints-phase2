# Tables 1-3 (regenerated)

# Table 1. Initial baseline results (frozen temporal test fold)

| Baseline | Definition | macro-F1 | 95% CI | Balanced acc. | Accuracy | s | MB |
|---|---|---|---|---|---|---|---|
| B0 | majority class | 0.010094 | [0.009869, 0.010544] | 0.019608 | 0.346609 | 0.0 | 117.2 |
| B1 | Product->Issue prior | 0.093778 | [0.088884, 0.099962] | 0.163208 | 0.471088 | 0.003 | 117.2 |
| B2 | bag-of-words Naive Bayes | 0.155597 | [0.140881, 0.172674] | 0.165756 | 0.390602 | 18.654 | 139.5 |
| B3 | LLM (frozen eval set) | 0.170558 | [0.128355, 0.192066] | 0.20887 | 0.1825 | 0 (cache replay) | - |

B3 is scored on a n=400 Issue-stratified subset of the same frozen test fold; B0-B2 use the full n=6,001 fold. The two sample frames are different, so Table 1 must not be read as a like-for-like ranking. `results/table1b_llm_subset.md` recomputes B0/B1/B2 on the identical 400 records.

# Table 2. LLM prompt and model configuration snapshot

| Item | Value |
|---|---|
| Config file | `configs/llm.json` (SHA256 `0df9558245d2362b...`) |
| Prompt | `prompts/v1.txt` (SHA256 `b1387736f441238b...`) |
| Few-shot | `prompts/few_shot.json` (4 TRAIN examples, 0 eval overlap) |
| Model | `WorkBuddy session LLM (platform-managed)` / version `2026-11-07 session; platform does not expose a pinned snapshot id` |
| Provider | `workbuddy-in-session-llm` |
| temperature | 0.0 |
| top_p | 1.0 |
| max_tokens | 300 |
| seed | 20261107 |
| repeats | 3 |
| eval n | 400 |
| External transmission authorised | False |
| Backend used for this pilot | `cache_replay` |
| Simulation | False |

Output schema: `issue_label`, `evidence_quote`, `confidence`, `abstain`.
Any output failing schema validation is recorded as ABSTAIN and counted in the failure taxonomy.

# Table 3. Frozen split and sample sizes

| fold | n | share | date_min | date_max | distinct issues | sha256 |
|---|---|---|---|---|---|---|
| train | 27977 | 0.6994 | 2015-03-19 | 2025-01-26 | 144 | ca55b07a8961158e |
| val | 6022 | 0.1505 | 2025-01-27 | 2025-07-10 | 65 | d5627bba3f253040 |
| test | 6001 | 0.15 | 2025-07-11 | 2026-06-12 | 73 | 18ebd9ca2dc2dda2 |
