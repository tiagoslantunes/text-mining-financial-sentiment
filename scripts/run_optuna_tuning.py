# -*- coding: utf-8 -*-
"""Optuna hyperparameter tuning of LightGBM on RoBERTa embeddings (Section 5.5).

Reproduces results/tables/optuna_best_params.json — the tuned configuration that
gives the best classical-ML result (LightGBM Tuned | RoBERTa, OOF F1-macro 0.8021).
TPE sampler + median pruner, 50 trials, objective = 5-fold stratified macro-F1
(seed 42), identical to the protocol in src/evaluation.py.

Run from the project root:
    python scripts/run_optuna_tuning.py            # 50 trials (~20 min)
    python scripts/run_optuna_tuning.py --trials 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
TAB = BASE / "results" / "tables"
SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()

    X = np.load(PROC / "X_roberta_train.npy")
    y = pd.read_csv(BASE / "data" / "raw" / "train.csv")["label"].to_numpy()

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 15, 255),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = LGBMClassifier(**params, class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1)
        return cross_val_score(model, X, y, cv=CV, scoring="f1_macro", n_jobs=-1).mean()

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    out = {"best_value": float(study.best_value), "best_params": study.best_params}
    (TAB / "optuna_best_params.json").write_text(json.dumps(out, indent=2))
    print(f"\nBest OOF F1-macro: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")
    print(f"Saved -> {TAB / 'optuna_best_params.json'}")


if __name__ == "__main__":
    main()
