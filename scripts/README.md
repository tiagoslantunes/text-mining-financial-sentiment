# Scripts

Run scripts from the repository root so their default relative paths resolve correctly.

## Canonical entry points

| Script | Purpose |
|---|---|
| `generate_features.py` | Build reusable sparse and dense feature artifacts |
| `features_analysis.py` | Compare engineered representations and classical models |
| `phase2_analysis.py` | Data-quality, encoding, shift, and noise analyses |
| `run_transformer_cv_v2.py` | Fine-tune one transformer with stratified cross-validation |
| `compare_and_ensemble.py` | Compare cached OOF predictions and build a soft-voting ensemble |
| `reoptimize_ensemble.py` | Coordinate-ascent optimisation of ensemble weights |
| `run_distill_cv.py` | Distil teacher probabilities into the final FinBERT student |
| `finalize_submission.py` | Synchronise the selected prediction artifact and final summaries |

`run_transformer_cv.py` is retained for the earlier experiment documented in the full notebook. New transformer runs should use `run_transformer_cv_v2.py`, which contains the final training safeguards and configuration options.

## Orchestration helpers

The PowerShell files reproduce specific experiment batches used during the project:

- `run_experiments_v3.ps1` — selected transformer experiment batch
- `run_all_10fold.ps1` — full 10-fold encoder batch
- `run_kd_grid.ps1` — knowledge-distillation hyperparameter grid

Review a script before launching it: the full transformer suite downloads multiple checkpoints and requires substantial GPU time and disk space.

Use `python <script> --help` for the Python entry points that expose CLI arguments. Exact final-model commands are recorded in [`../docs/reproducibility.md`](../docs/reproducibility.md).
