# Table 1b. Same-frame control on the LLM evaluation subset

All four rows use the SAME 400 Issue-stratified frozen test records and the same redacted, 800-character-truncated input. Paired bootstrap 95% CI, 500 resamples, seed 20261107.

| Method | macro-F1 | 95% CI | Balanced acc. | Accuracy | Abstention |
|---|---|---|---|---|---|
| B3 LLM | 0.170558 | [0.128355, 0.192066] | 0.20887 | 0.1825 | 0.0 |
| B0-sub | 0.000674 | [0.000292, 0.001165] | 0.019608 | 0.0175 | 0.0 |
| B1-sub | 0.066157 | [0.053969, 0.077294] | 0.167727 | 0.2425 | 0.0 |
| B2-sub | 0.113823 | [0.082045, 0.136547] | 0.16432 | 0.1625 | 0.0 |

Table 1's B0-B2 rows use the full n=6,001 test fold and must not be compared directly with these subset figures.
