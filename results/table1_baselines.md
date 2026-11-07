# Table 1. Initial baseline results (frozen temporal test fold)

| Baseline | Definition | macro-F1 | 95% CI | Balanced acc. | Accuracy | s | MB |
|---|---|---|---|---|---|---|---|
| B0 | majority class | 0.010094 | [0.009869, 0.010544] | 0.019608 | 0.346609 | 0.0 | 117.2 |
| B1 | Product->Issue prior | 0.093778 | [0.088884, 0.099962] | 0.163208 | 0.471088 | 0.003 | 117.2 |
| B2 | bag-of-words Naive Bayes | 0.155597 | [0.140881, 0.172674] | 0.165756 | 0.390602 | 18.654 | 139.5 |
| B3 | LLM (frozen eval set) | 0.170558 | [0.128355, 0.192066] | 0.20887 | 0.1825 | 0 (cache replay) | - |

B3 is scored on a n=400 Issue-stratified subset of the same frozen test fold; B0-B2 use the full n=6,001 fold. The two sample frames are different, so Table 1 must not be read as a like-for-like ranking. `results/table1b_llm_subset.md` recomputes B0/B1/B2 on the identical 400 records.
