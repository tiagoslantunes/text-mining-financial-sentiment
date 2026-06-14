# -*- coding: utf-8 -*-
"""Train and evaluate the traditional ML models (Section 5.1) and save the
comparison table to results/tables/classical_ml_comparison.csv.

This is the script that produces the classical-ML half of the model comparison:
every (model family x feature representation) combination is evaluated with the
shared 5-fold stratified protocol from src/evaluation.py (seed 42), exactly as
documented in the report.

Model families  : MultinomialNB, ComplementNB, LogisticRegression, LinearSVC,
                  KNN, MLP, RandomForest, XGBoost, LightGBM (+ Optuna-tuned).
Feature sets    : sparse  (BoW, TF-IDF 1g/2g/char)   -> NB, LR, SVM
                  static  (Word2Vec SG/CBOW, GloVe)  -> dense models
                  encoders (FinBERT, SBERT, RoBERTa) -> dense models

Run from the project root:
    python scripts/run_classical_ml.py            # full grid (~30-40 min)
    python scripts/run_classical_ml.py --quick    # a fast representative subset
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(BASE))
from src.evaluation import evaluate_model   # shared 5-fold protocol (seed 42)

PROC = BASE / "data" / "processed"
TAB = BASE / "results" / "tables"
SEED = 42


def load_features(quick=False):
    """Return {name: X} for every representation. Sparse are fit here from raw text."""
    train = pd.read_csv(BASE / "data" / "raw" / "train.csv")
    texts = train["text"].astype(str).tolist()
    y = train["label"].to_numpy()

    sparse = {
        "BoW":         CountVectorizer(binary=True, max_features=20000, min_df=2),
        "TF-IDF 1g":   TfidfVectorizer(ngram_range=(1, 1), max_features=20000, sublinear_tf=True, min_df=2, max_df=0.8),
        "TF-IDF 2g":   TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True, min_df=2, max_df=0.8),
        "TF-IDF char": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000, sublinear_tf=True),
    }
    sparse_X = {name: vec.fit_transform(texts) for name, vec in sparse.items()}

    dense_files = {
        "W2V-SG":        "X_w2v_sg_train.npy",
        "W2V-CBOW":      "X_w2v_cbow_train.npy",
        "GloVe-Twitter": "X_glove_train.npy",
        "FinBERT":       "X_finbert_train.npy",
        "SBERT":         "X_sbert_train.npy",
        "RoBERTa":       "X_roberta_train.npy",
    }
    dense_X = {}
    for name, f in dense_files.items():
        p = PROC / f
        if p.exists():
            dense_X[name] = np.load(p)
        else:
            print(f"  [skip dense] {name}: {f} not found (run generate_features.py)")
    if quick:
        sparse_X = {k: sparse_X[k] for k in ["TF-IDF 2g"]}
        dense_X = {k: dense_X[k] for k in ["SBERT", "RoBERTa"] if k in dense_X}
    return sparse_X, dense_X, y


def model_zoo(quick=False):
    """Return (sparse_models, dense_models) — name -> estimator factory."""
    try:
        tuned = json.loads((TAB / "optuna_best_params.json").read_text())["best_params"]
    except Exception:
        tuned = {}

    sparse_models = {
        "MultinomialNB a=1.0":  lambda: MultinomialNB(alpha=1.0),
        "ComplementNB a=0.5":   lambda: ComplementNB(alpha=0.5),
        "LR C=0.1":             lambda: LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=SEED),
        "LR C=1.0":             lambda: LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED),
        "LR C=10.0":            lambda: LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced", random_state=SEED),
        "LinearSVC C=0.1":      lambda: CalibratedClassifierCV(LinearSVC(C=0.1, class_weight="balanced", random_state=SEED)),
        "LinearSVC C=1.0":      lambda: CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)),
    }
    dense_models = {
        "LR C=1.0":        lambda: LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED),
        "KNN k=5":         lambda: KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "KNN k=15":        lambda: KNeighborsClassifier(n_neighbors=15, metric="cosine"),
        "MLP (256,128)":   lambda: MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True, random_state=SEED),
        "RF 100":          lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "RF 300":          lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "XGBoost 100":     lambda: XGBClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, verbosity=0),
        "XGBoost 300":     lambda: XGBClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, verbosity=0),
        "LightGBM 100":    lambda: LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1),
        "LightGBM 300":    lambda: LGBMClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1),
    }
    if tuned:
        dense_models["LightGBM Tuned"] = lambda: LGBMClassifier(**tuned, class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1)
    if quick:
        sparse_models = {k: sparse_models[k] for k in ["LR C=1.0"]}
        dense_models = {k: dense_models[k] for k in ["LightGBM 300", "LightGBM Tuned"] if k in dense_models}
    return sparse_models, dense_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast representative subset")
    args = ap.parse_args()

    print("Loading features ...")
    sparse_X, dense_X, y = load_features(args.quick)
    sparse_models, dense_models = model_zoo(args.quick)

    rows = []
    t0 = time.time()

    # sparse models x sparse features
    for fname, X in sparse_X.items():
        for mname, factory in sparse_models.items():
            r = evaluate_model(factory(), X, y, f"{mname} | {fname}")
            rows.append(r)
            print(f"  {r['Model']:<32} F1={r['F1-macro']:.4f}")

    # LR also on dense features (it is the only sparse model that handles them)
    for fname, X in dense_X.items():
        r = evaluate_model(LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED),
                           X, y, f"LR C=1.0 | {fname}")
        rows.append(r)
        print(f"  {r['Model']:<32} F1={r['F1-macro']:.4f}")

    # dense models x dense features
    for fname, X in dense_X.items():
        for mname, factory in dense_models.items():
            if mname == "LR C=1.0":
                continue
            r = evaluate_model(factory(), X, y, f"{mname} | {fname}")
            rows.append(r)
            print(f"  {r['Model']:<32} F1={r['F1-macro']:.4f}")

    df = pd.DataFrame(rows).sort_values("F1-macro", ascending=False).reset_index(drop=True)
    out = TAB / "classical_ml_comparison.csv"
    df.to_csv(out, index=True)
    print(f"\n{len(df)} models evaluated in {(time.time()-t0)/60:.1f} min -> {out}")
    print("\nTop 10:")
    print(df[["Model", "F1-macro", "Accuracy"]].head(10).to_string())


if __name__ == "__main__":
    main()
