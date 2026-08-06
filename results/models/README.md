# Model Artifacts

This directory preserves the two lightweight artifacts required by the offline agent notebook:

| File | Purpose |
|---|---|
| `agent_lgbm.pkl` | LightGBM classifier trained on the agent's 80% training partition |
| `agent_holdout_idx.npy` | Indices for the untouched 20% calibration/evaluation holdout |

Large transformer checkpoints and other regenerable models are intentionally excluded. The agent downloads its public SBERT and FinBERT dependencies on first use.

`agent_lgbm.pkl` is a Python pickle/joblib artifact. Pickle files can execute code during deserialization; load this file only from a trusted, integrity-checked checkout of this repository. Do not replace it with an untrusted download.
