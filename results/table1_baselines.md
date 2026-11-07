# Table 1. Initial baseline results (frozen temporal test fold)

Target: `Issue`; input: `Consumer complaint narrative`; n_train=27977, n_val=6022, n_test=6001; seed=20261107; bootstrap=500.

| Baseline | Definition | macro-F1 | 95% CI | Balanced acc. | Accuracy | s | MB |
|---|---|---|---|---|---|---|---|
| B0 | majority class | 0.010094 | [0.009869, 0.010544] | 0.019608 | 0.346609 | 0.0 | 117.7 |
| B1 | Product->Issue prior | 0.093778 | [0.088884, 0.099962] | 0.163208 | 0.471088 | 0.003 | 117.7 |
| B2 | bag-of-words Naive Bayes | 0.155597 | [0.140881, 0.172674] | 0.165756 | 0.390602 | 19.931 | 139.4 |
| B3 | LLM (frozen eval set) | NOT RUN | NOT RUN | NOT RUN | NOT RUN | - | - |

B3 (LLM) is intentionally unscored: no authorised endpoint was available. See `src/llm_eval.py` and section 7.5 of the report.
