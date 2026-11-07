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
