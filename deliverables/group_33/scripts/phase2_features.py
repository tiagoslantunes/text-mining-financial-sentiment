"""Phase 2 part B: preprocessing ablation (fix_text on TF-IDF), fusion features,
encoder PCA visualisation, SMOTE comparison.

Run: python scripts/phase2_features.py
"""

import json
import re
import time
from pathlib import Path

import ftfy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / "results" / "figures"
TAB = BASE / "results" / "tables"
PROC = BASE / "data" / "processed"

train = pd.read_csv(BASE / "data" / "raw" / "train.csv")
y = train["label"].values
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

_TRUNC_RE = re.compile(r'[�…°]+\s*(https?://\S*)?$')
_URL_RE = re.compile(r'https?://\S+')
_TRAIL_RE = re.compile(r'[\s\-–:]+$')

def fix_tweet(text):
    text = ftfy.fix_text(str(text))
    text = _TRUNC_RE.sub('', text)
    text = _URL_RE.sub('', text)
    return _TRAIL_RE.sub('', text).strip()

# ============================================================
# 4.2  Preprocessing ablation: raw vs raw+fix_text (LR + TF-IDF)
# ============================================================
print("=" * 60)
print("4.2  fix_text ablation com LR + TF-IDF 10k")
print("=" * 60)

ablation_rows = []
for name, texts in [("raw", train["text"]),
                    ("raw + fix_text", train["text"].apply(fix_tweet))]:
    vec = TfidfVectorizer(max_features=10000, sublinear_tf=True)
    X = vec.fit_transform(texts.astype(str))
    scores = cross_val_score(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        X, y, cv=CV, scoring="f1_macro", n_jobs=-1)
    print(f"{name}: F1-macro={scores.mean():.4f} +/- {scores.std():.4f}")
    ablation_rows.append({"Config": name, "F1-macro": round(scores.mean(), 4),
                          "Std": round(scores.std(), 4)})

# Merge into existing ablation table
abl_path = TAB / "preprocessing_ablation.csv"
if abl_path.exists():
    existing = pd.read_csv(abl_path)
    print(f"\nTabela existente ({len(existing)} configs):")
    print(existing.to_string(index=False))
    # Add fix_text rows if not present
    config_col = existing.columns[0]
    for row in ablation_rows:
        if row["Config"] not in existing[config_col].values:
            new_row = {c: None for c in existing.columns}
            new_row[config_col] = row["Config"]
            # find F1 column
            f1_cols = [c for c in existing.columns if "f1" in c.lower() or "F1" in c]
            if f1_cols:
                new_row[f1_cols[0]] = row["F1-macro"]
            existing = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    existing.to_csv(abl_path, index=False)
    print(f"\nTabela actualizada ({len(existing)} configs)")
else:
    pd.DataFrame(ablation_rows).to_csv(abl_path, index=False)

# ============================================================
# 4.3-A  Fusion: SBERT + financial features
# ============================================================
print()
print("=" * 60)
print("4.3-A  Fusion SBERT + financial (LightGBM)")
print("=" * 60)

from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

X_sbert = np.load(PROC / "X_sbert_train.npy")
X_fin = np.load(PROC / "X_fin_train.npy")
scaler = StandardScaler()
X_fin_scaled = scaler.fit_transform(X_fin)
X_fusion = np.hstack([X_sbert, X_fin_scaled])
print(f"Fusion shape: {X_fusion.shape}")

best_params = json.loads((TAB / "optuna_best_params.json").read_text())["best_params"]
lgbm = LGBMClassifier(**best_params, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
t0 = time.time()
scores_fusion = cross_val_score(lgbm, X_fusion, y, cv=CV, scoring="f1_macro")
print(f"LightGBM (SBERT+financial fusion): F1={scores_fusion.mean():.4f} +/- {scores_fusion.std():.4f} ({time.time()-t0:.0f}s)")

lgbm2 = LGBMClassifier(**best_params, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
scores_sbert = cross_val_score(lgbm2, X_sbert, y, cv=CV, scoring="f1_macro")
print(f"LightGBM (SBERT only):             F1={scores_sbert.mean():.4f} +/- {scores_sbert.std():.4f}")

fusion_result = {
    "sbert_only_f1": round(float(scores_sbert.mean()), 4),
    "fusion_f1": round(float(scores_fusion.mean()), 4),
    "gain": round(float(scores_fusion.mean() - scores_sbert.mean()), 4),
}
(TAB / "fusion_features_result.json").write_text(json.dumps(fusion_result, indent=2))
print(f"Fusion gain: {fusion_result['gain']:+.4f}")

# ============================================================
# 4.3-B  PCA 2D of the three frozen encoders
# ============================================================
print()
print("=" * 60)
print("4.3-B  Encoder PCA 2D")
print("=" * 60)

from sklearn.decomposition import PCA

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, fname) in zip(axes, [
        ("FinBERT CLS (frozen)", "X_finbert_train.npy"),
        ("SBERT all-mpnet", "X_sbert_train.npy"),
        ("Twitter-RoBERTa", "X_roberta_train.npy")]):
    X = np.load(PROC / fname)
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(X)
    for label, lname, color in [(0, "Bearish", "#d62728"), (1, "Bullish", "#2ca02c"), (2, "Neutral", "#1f77b4")]:
        m = y == label
        ax.scatter(X2d[m, 0], X2d[m, 1], c=color, label=lname, alpha=0.3, s=4)
    ax.set_title(f"{name}\nVar explained: {pca.explained_variance_ratio_.sum():.1%}")
    ax.legend(fontsize=8, markerscale=3)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(FIG / "encoder_pca.pdf", dpi=300, bbox_inches="tight")
plt.savefig(FIG / "encoder_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved encoder_pca.{pdf,png}")

# ============================================================
# 4.4  SMOTE vs class_weight (LightGBM + SBERT)
# ============================================================
print()
print("=" * 60)
print("4.4  SMOTE vs class_weight")
print("=" * 60)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

lgbm_cw = LGBMClassifier(**best_params, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
f1_cw = cross_val_score(lgbm_cw, X_sbert, y, cv=CV, scoring="f1_macro").mean()

smote_pipe = ImbPipeline([
    ("smote", SMOTE(random_state=42, k_neighbors=5)),
    ("clf", LGBMClassifier(**best_params, random_state=42, n_jobs=-1, verbose=-1)),
])
f1_smote = cross_val_score(smote_pipe, X_sbert, y, cv=CV, scoring="f1_macro").mean()

print(f"LightGBM SBERT + class_weight: {f1_cw:.4f}")
print(f"LightGBM SBERT + SMOTE:        {f1_smote:.4f}")
smote_result = {
    "class_weight_f1": round(float(f1_cw), 4),
    "smote_f1": round(float(f1_smote), 4),
    "delta": round(float(f1_smote - f1_cw), 4),
    "conclusion": "SMOTE melhor" if f1_smote > f1_cw + 0.005 else "class_weight suficiente (SMOTE descartado)",
}
(TAB / "smote_comparison.json").write_text(json.dumps(smote_result, indent=2))
print(f"Conclusao: {smote_result['conclusion']}")

# Checkpoint
cp = BASE / "results" / "progress_checkpoint.json"
state = json.loads(cp.read_text()) if cp.exists() else {}
state["phase_2_features"] = {"done": True, "timestamp": time.time(), **fusion_result, **smote_result}
cp.write_text(json.dumps(state, indent=2))
print("\nCheckpoint phase_2_features guardado.")
