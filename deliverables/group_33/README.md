# Group 33 — Financial Tweet Sentiment Classification
Text Mining 2025/2026 · NOVA IMS

## Deliverables

| File | Description |
|---|---|
| `pred_33.csv` | Test predictions (`id`, `label`) produced by the final pipeline |
| `tm_final_33.ipynb` | Final solution — a single integrated **weighted soft-vote ensemble** pipeline (out-of-fold F1-macro 0.9201). Restart & Run All regenerates `pred_33.csv` in ~2 min using the committed per-model probabilities, or ~40 min on a GPU if the primary encoder is retrained from scratch. |
| `tm_tests_33.ipynb` | All experiments and their evaluation: data exploration, preprocessing ablations, feature engineering, traditional ML and transformer families, backbone search, ensemble, knowledge distillation, and statistical tests. Ships fully executed. |
| `report_33.pdf` | Project report. |
| `notebooks/08_Agentic_Workflow.ipynb` | Conversational agentic workflow — runs fully offline with open-source models only. |

## Environment

```bash
pip install -r requirements.txt   # Python 3.11
```

A CUDA GPU accelerates training but is **not** required: the per-model probabilities are
shipped, so `tm_final_33.ipynb` and the notebooks run on CPU. An internet connection is
needed on first run to download the public Hugging Face models and NLTK corpora (anonymous,
no account or token).

## Reproducing the artefacts from scratch

| Step | Command |
|---|---|
| Build feature representations | `python scripts/generate_features.py` |
| Train the 8 transformer encoders | `scripts/run_all_10fold.ps1` |
| Grid-search the ensemble weights | `python scripts/reoptimize_ensemble.py` |
| Feature/EDA analyses (ablations, fusion, PCA, SMOTE, calibration, significance) | `python scripts/feature_analyses.py` and `python scripts/eda_analyses.py` |

All randomness is seeded (`seed=42`) and every transformer result uses an identical
10-fold `StratifiedKFold`.

## Notes

- **Open-source models only** — no proprietary APIs and no API keys anywhere in the project.
- The 153 MB of dense embeddings in `data/processed/` are not bundled; the feature-loading
  cell in `tm_tests_33.ipynb` regenerates them on demand via `scripts/generate_features.py`.
