# Table 2. LLM prompt and model configuration snapshot

| Item | Value |
|---|---|
| Config file | `configs/llm.json` (SHA256 `721e1beb9cdc2d15...`) |
| Prompt | `prompts/v1.txt` (SHA256 `b1387736f441238b...`) |
| Few-shot | `prompts/few_shot.json` (4 TRAIN examples, 0 eval overlap) |
| Model | `gpt-4o-mini` / version `PIN_EXACT_SNAPSHOT_AFTER_AUTHORISATION` |
| Provider | `openai-compatible` |
| temperature | 0.0 |
| top_p | 1.0 |
| max_tokens | 300 |
| seed | 20261107 |
| repeats | 3 |
| eval n | 400 |
| External transmission authorised | False |
| Backend used for this pilot | `mock_deterministic` |
| Simulation | True |

Output schema: `issue_label`, `evidence_quote`, `confidence`, `abstain`.
Any output failing schema validation is recorded as ABSTAIN and counted in the failure taxonomy.
