# -*- coding: utf-8 -*-
"""Stacked generalisation (Wolpert, 1992) — extra work. Reproduces the stacking
result in results/tables/phase6_fusion_stacking.csv.

Base learners emit out-of-fold (OOF) class probabilities; these are concatenated
into a meta-feature matrix on which a Logistic-Regression meta-learner is trained
under the shared 5-fold protocol. Because every base learner uses the identical
fold split, the meta-features are leakage-free.

Base learners are read from results/predictions/oof_proba_*.npy (each produced by
run_classical_ml.py / run_transformer_cv.py). Pass --models to choose which.

Run from the project root:
    python scripts/run_stacking.py
    python scripts/run_stacking.py --models finbert_fintwitter_10ep_fixtext roberta_large_ts_v2 ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score

BASE = Path(__file__).resolve().parent.parent
PRED = BASE / "results" / "predictions"
TAB = BASE / "results" / "tables"
SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

DEFAULT_BASE = [
    "finbert_fintwitter_10ep_fixtext",
    "finbert_fintwitter_7ep",
    "roberta_large_ts_v2",
    "debertav3_large_6ep",
    "deberta_base_finance_fixtext",
]


def load_oof(tag):
    for p in [PRED / f"oof_proba_{tag}.npy", PRED / f"oof_proba_{tag}.csv"]:
        if p.exists():
            return (np.load(p) if p.suffix == ".npy"
                    else pd.read_csv(p)[["p0", "p1", "p2"]].values).astype(np.float64)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_BASE)
    args = ap.parse_args()

    y = pd.read_csv(BASE / "data" / "raw" / "train.csv")["label"].to_numpy()

    blocks, used = [], []
    for tag in args.models:
        oof = load_oof(tag)
        if oof is not None:
            blocks.append(oof)
            used.append(tag)
        else:
            print(f"  [skip] {tag}: no OOF probabilities found")
    if len(blocks) < 2:
        raise SystemExit("Need >=2 base learners with OOF probabilities.")

    meta_X = np.hstack(blocks)   # (N, 3*n_base) leakage-free meta-features
    print(f"Stacking {len(used)} base learners -> meta-features {meta_X.shape}")

    # meta-learner: Logistic Regression, evaluated with the same 5-fold protocol
    meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    meta_pred = cross_val_predict(meta, meta_X, y, cv=CV, method="predict")
    f1 = f1_score(y, meta_pred, average="macro")
    acc = accuracy_score(y, meta_pred)

    # soft-average reference
    avg_pred = np.mean(blocks, axis=0).argmax(1)
    f1_avg = f1_score(y, avg_pred, average="macro")

    print(f"\nStacking (LR meta)  : F1-macro = {f1:.4f}  | accuracy = {acc:.4f}")
    print(f"Soft-Average        : F1-macro = {f1_avg:.4f}")

    rows = [
        {"Model": f"Stacking (LR meta) | {len(used)} base learners", "F1-macro": round(float(f1), 4), "Accuracy": round(float(acc), 4)},
        {"Model": f"Soft-Average | {len(used)} base learners",       "F1-macro": round(float(f1_avg), 4)},
    ]
    pd.DataFrame(rows).to_csv(TAB / "stacking_comparison.csv", index=False)
    (TAB / "stacking_summary.json").write_text(json.dumps(
        {"base_learners": used, "stacking_f1": float(f1), "soft_average_f1": float(f1_avg)}, indent=2))
    print(f"Saved -> {TAB / 'stacking_comparison.csv'}")


if __name__ == "__main__":
    main()
