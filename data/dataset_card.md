# Dataset Card — CFPB Consumer Complaint Database (Phase 2 snapshot)

## Motivation

The project studies whether consumer-complaint issue labels can be predicted from the narrative a consumer submits, and whether a large language model adds value over simple baselines under a leakage-free evaluation design.

## Composition

- Observation unit: one complaint (`Complaint ID`). **16,545,376** logical records, **16** columns.
- Physical lines **27,656,645** > logical records: narratives contain quoted newlines; count logical CSV records, never physical lines.
- Date range `2011-12-01` to `2026-07-05` (176 calendar months).
- Target `Issue`: 180 classes. Features: submission-time fields only.
- Sample frame for text modelling: **3,814,581** records with non-empty narrative and Issue; seeded reservoir of **40,000** is used for the baselines.

## Collection process

- Source: Consumer Financial Protection Bureau (CFPB), Consumer Complaint Database, public download of `complaints.csv.zip` containing `complaints.csv`. Official description and terms: https://www.consumerfinance.gov/data-research/consumer-complaints/ [1].
- Licence / ownership: CFPB states that all complaint data it publishes is freely available for anyone to use, analyse and build on. CFPB also states the database is not a statistically representative sample of consumers' experiences in the marketplace; this is recorded as a scope limitation.
- Snapshot identity is fixed by the SHA256 of the uncompressed CSV (above), not by a download date: any new download with a different hash requires a fresh split and a full rerun. The archive was supplied with the course materials; no separate download event is claimed.
- Included population: complaints submitted to CFPB and published in the database. Excluded: consumers who never complained, unpublished records, non-US jurisdictions.

## Preprocessing

- Single-pass streaming decode with `newline=''`; SHA256 of the uncompressed CSV: `ee76ed0a27d33732d3c292de7902fd7a3fb9b07eb93ba3f69894f556dbd096c1`.
- ZIP privacy masks kept as masks; post-hoc fields separated from features; temporal 70/15/15 split frozen as SHA256-listed ID files.
- Narratives normalised, truncated to 2500 chars, and PII-redacted before any external model call.

## Uses

- In scope: data-quality profiling, issue-label classification, baseline and LLM comparison under a fixed evaluation protocol.
- Out of scope: predicting company responses from submission-time data, generating replies, causal claims about company behaviour, individual-level decisions about consumers.

## Distribution and maintenance

- The raw 8.9 GB file is NOT redistributed in this repository. Access method: the public CFPB Consumer Complaint Database download page [1]. The local plaintext copy stays in the team workspace, is excluded by `.gitignore`, and is not uploaded to any code host or third-party service.
- A small synthetic sample (`data/sample_complaints.csv`) is included for tests.
- Versioning: dataset identity is the uncompressed CSV SHA256 above; any new download requires a new split and a new run of `src/run_all.py`.

## Ethical considerations

- Narratives are de-identified but may still contain names, account numbers and addresses; no raw narrative is published in the report.
- `Tags` (servicemember / older American) and ZIP are sensitive; used only for subgroup auditing, never as predictors.
- Access control and deletion: the single local plaintext archive is not shared, not committed, and not sent to any external API. Only SHA256-pinned IDs, aggregate statistics, short redacted evidence quotes and derived artefacts are published. The archive can be deleted after reproduction; the committed results remain verifiable from the frozen split IDs and the response cache.
