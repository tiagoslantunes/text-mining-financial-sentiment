"""Compare all experiment results and optionally create a soft-vote ensemble.

Usage:
    python scripts/compare_and_ensemble.py                  # just compare
    python scripts/compare_and_ensemble.py --ensemble       # compare + ensemble
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def load_results(table_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(table_dir.glob("*_result.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            rows.append({
                "tag": data.get("tag", f.stem),
                "model_id": data.get("model_id", "?"),
                "F1-macro": round(data.get("F1-macro", 0), 4),
                "F1-weighted": round(data.get("F1-weighted", 0), 4),
                "Accuracy": round(data.get("Accuracy", 0), 4),
                "epochs": data.get("recipe", {}).get("epochs", "?"),
                "lr": data.get("recipe", {}).get("lr", "?"),
                "label_smoothing": data.get("recipe", {}).get("label_smoothing", "?"),
                "schedule": data.get("recipe", {}).get("schedule", "?"),
            })
        except Exception as e:
            print(f"Could not read {f}: {e}")
    df = pd.DataFrame(rows).sort_values("F1-macro", ascending=False)
    return df


def try_ensemble(pred_dir: Path, table_dir: Path, tags: list[str], train_csv: str, test_csv: str):
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    y = train["label"].to_numpy()

    oof_stacks = []
    test_stacks = []
    used = []
    for tag in tags:
        oof_path = pred_dir / f"oof_proba_{tag}.npy"
        test_path = pred_dir / f"test_proba_{tag}.npy"
        if oof_path.exists() and test_path.exists():
            oof_stacks.append(np.load(oof_path))
            test_stacks.append(np.load(test_path))
            used.append(tag)
        else:
            print(f"  Missing predictions for: {tag}")

    if len(used) < 2:
        print("Need at least 2 models for ensemble.")
        return

    print(f"\nEnsembling: {used}")
    oof_avg = np.mean(oof_stacks, axis=0)
    test_avg = np.mean(test_stacks, axis=0)

    oof_f1 = f1_score(y, oof_avg.argmax(axis=1), average="macro")
    print(f"Soft-average ensemble F1-macro: {oof_f1:.6f}")

    # Optionally try weighted ensembles
    individual_f1s = []
    for tag, oof in zip(used, oof_stacks):
        f1 = f1_score(y, oof.argmax(axis=1), average="macro")
        individual_f1s.append(f1)
        print(f"  {tag}: {f1:.6f}")

    # Weighted by individual F1
    weights = np.array(individual_f1s)
    weights = weights / weights.sum()
    oof_weighted = np.average(oof_stacks, axis=0, weights=weights)
    test_weighted = np.average(test_stacks, axis=0, weights=weights)
    wf1 = f1_score(y, oof_weighted.argmax(axis=1), average="macro")
    print(f"F1-weighted ensemble (by individual score): {wf1:.6f}")

    best_oof = oof_weighted if wf1 > oof_f1 else oof_avg
    best_test = test_weighted if wf1 > oof_f1 else test_avg
    best_f1 = max(wf1, oof_f1)
    method = "weighted" if wf1 > oof_f1 else "equal"

    ensemble_tag = "ensemble_" + "_".join(used)[:60]
    test_pred = best_test.argmax(axis=1)
    pd.DataFrame({"id": test["id"], "label": test_pred}).to_csv(
        pred_dir / f"pred_{ensemble_tag}.csv", index=False
    )
    print(f"\nBest ensemble ({method}): F1={best_f1:.6f}")
    print(f"Saved to: pred_{ensemble_tag}.csv")
    print(f"Dist: {np.bincount(test_pred, minlength=3).tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--top-n", type=int, default=3, help="Top N models for ensemble")
    args = parser.parse_args()

    table_dir = Path(args.out_dir) / "tables"
    pred_dir = Path(args.out_dir) / "predictions"

    df = load_results(table_dir)
    print("\n=== MODEL COMPARISON ===")
    print(df.to_string(index=False))

    if args.ensemble and len(df) >= 2:
        top_tags = df.head(args.top_n)["tag"].tolist()
        try_ensemble(pred_dir, table_dir, top_tags, args.train_csv, args.test_csv)


if __name__ == "__main__":
    main()
