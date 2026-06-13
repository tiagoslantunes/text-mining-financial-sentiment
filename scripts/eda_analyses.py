"""Phase 2 analyses: EDA fix_text, VADER by class, transformer ablation,
calibration, statistical significance. All outputs go to results/.

Run: python scripts/eda_analyses.py
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
from sklearn.metrics import f1_score

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / "results" / "figures"
TAB = BASE / "results" / "tables"
PRED = BASE / "results" / "predictions"

train = pd.read_csv(BASE / "data" / "raw" / "train.csv")
y = train["label"].values

# ============================================================
# 4.1-A  Mojibake analysis
# ============================================================
print("=" * 60)
print("4.1-A  Mojibake / fix_text analysis")
print("=" * 60)

def has_mojibake(text):
    t = str(text)
    return ftfy.fix_text(t) != t or "�" in t

mask_moji = train["text"].apply(has_mojibake)
n_corrupted = int(mask_moji.sum())
pct = 100 * n_corrupted / len(train)
print(f"Tweets com mojibake/encoding artefacts: {n_corrupted} ({pct:.1f}%)")

# Per-class breakdown
moji_by_class = train[mask_moji]["label"].value_counts().sort_index()
total_by_class = train["label"].value_counts().sort_index()
rows = []
for label, name in [(0, "Bearish"), (1, "Bullish"), (2, "Neutral")]:
    n = int(moji_by_class.get(label, 0))
    tot = int(total_by_class[label])
    rows.append({"Class": name, "Corrupted": n, "Total": tot, "Pct": 100 * n / tot})
moji_table = pd.DataFrame(rows)
moji_table.to_csv(TAB / "mojibake_by_class.csv", index=False)
print(moji_table.to_string(index=False))

# Example pairs (before/after)
examples = []
for label in [0, 1, 2]:
    sel = train[(train["label"] == label) & mask_moji].head(2)
    for _, row in sel.iterrows():
        fixed = ftfy.fix_text(row["text"])
        if fixed != row["text"]:
            examples.append({"label": label, "before": row["text"][:100], "after": fixed[:100]})
pd.DataFrame(examples).to_csv(TAB / "mojibake_examples.csv", index=False)
print(f"Guardados {len(examples)} exemplos before/after")

# ============================================================
# 4.1-B  VADER by class
# ============================================================
print()
print("=" * 60)
print("4.1-B  VADER compound by class")
print("=" * 60)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
train["vader_compound"] = train["text"].apply(lambda t: sia.polarity_scores(str(t))["compound"])

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (label, name, color) in zip(axes, [(0, "Bearish", "#d62728"), (1, "Bullish", "#2ca02c"), (2, "Neutral", "#1f77b4")]):
    data = train[train["label"] == label]["vader_compound"]
    ax.hist(data, bins=30, color=color, alpha=0.8, edgecolor="white")
    ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.5, label=f"mean={data.mean():.2f}")
    ax.set_title(f"{name} - VADER compound")
    ax.set_xlabel("Compound score")
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "vader_by_class.pdf", dpi=300, bbox_inches="tight")
plt.savefig(FIG / "vader_by_class.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved vader_by_class.{pdf,png}")

# 4.1-C Neutral ambiguity table
neutral = train[train["label"] == 2]
vpos = int((neutral["vader_compound"] > 0.05).sum())
vneg = int((neutral["vader_compound"] < -0.05).sum())
vneu = len(neutral) - vpos - vneg
amb = {
    "neutral_total": len(neutral),
    "vader_positive": vpos, "vader_positive_pct": round(100 * vpos / len(neutral), 1),
    "vader_negative": vneg, "vader_negative_pct": round(100 * vneg / len(neutral), 1),
    "vader_neutral": vneu, "vader_neutral_pct": round(100 * vneu / len(neutral), 1),
}
(TAB / "vader_neutral_ambiguity.json").write_text(json.dumps(amb, indent=2))
print(f"Neutral tweets com VADER positivo: {vpos} ({amb['vader_positive_pct']}%), "
      f"negativo: {vneg} ({amb['vader_negative_pct']}%), neutro: {vneu} ({amb['vader_neutral_pct']}%)")

# ============================================================
# 4.5  Transformer ablation table (fix_text impact)
# ============================================================
print()
print("=" * 60)
print("4.5  Transformer fix_text ablation")
print("=" * 60)

pairs = [
    ("finbert_fintwitter_10ep", "finbert_fintwitter_10ep_fixtext", "FinBERT-fintwitter 10ep"),
    ("debertav3_large_6ep", "debertav3_large_6ep_fixtext", "DeBERTa-v3-large 6ep"),
]
rows = []
for tag_raw, tag_fix, name in pairs:
    for tag, variant in [(tag_raw, "raw text"), (tag_fix, "fix_text")]:
        p = TAB / f"{tag}_result.json"
        if p.exists():
            d = json.loads(p.read_text())
            rows.append({"Model": name, "Preprocessing": variant,
                         "OOF F1-macro": round(d["F1-macro"], 4),
                         "Accuracy": round(d["Accuracy"], 4)})
ablation_tx = pd.DataFrame(rows)
ablation_tx["Delta F1"] = ablation_tx.groupby("Model")["OOF F1-macro"].diff().round(4)
ablation_tx.to_csv(TAB / "preprocessing_ablation_transformers.csv", index=False)
print(ablation_tx.to_string(index=False))

# ============================================================
# 4.6  Calibration analysis + temperature scaling
# ============================================================
print()
print("=" * 60)
print("4.6  Calibration analysis")
print("=" * 60)

def apply_temperature(proba, T):
    logits = np.log(proba + 1e-9)
    scaled = logits / T
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)

cal_results = {}
for tag in ["finbert_fintwitter_10ep_fixtext", "finbert_fintwitter_7ep", "debertav3_large_6ep"]:
    oof = np.load(PRED / f"oof_proba_{tag}.npy")
    max_proba = oof.max(axis=1)
    pred = oof.argmax(axis=1)
    correct = (pred == y)

    df_cal = pd.DataFrame({"confidence": max_proba, "correct": correct})
    df_cal["bucket"] = pd.cut(df_cal["confidence"], bins=[0, .5, .6, .7, .8, .9, 1.0])
    cal_table = df_cal.groupby("bucket", observed=True)["correct"].agg(["mean", "count"])
    print(f"\n{tag} - calibration (accuracy vs confidence bucket):")
    print(cal_table.to_string())

    # Temperature grid search on OOF
    best_T, best_f1 = 1.0, f1_score(y, pred, average="macro")
    base_f1 = best_f1
    for T in [0.5, 0.7, 0.9, 1.0, 1.2, 1.5]:
        cal = apply_temperature(oof, T)
        f1 = f1_score(y, cal.argmax(1), average="macro")
        if f1 > best_f1:
            best_T, best_f1 = T, f1
    cal_results[tag] = {"base_f1": round(float(base_f1), 4),
                        "best_T": best_T,
                        "best_f1": round(float(best_f1), 4),
                        "gain": round(float(best_f1 - base_f1), 4)}
    print(f"  Temperature search: best T={best_T}, F1={best_f1:.4f} (gain {best_f1-base_f1:+.4f})")

(TAB / "calibration_results.json").write_text(json.dumps(cal_results, indent=2))
print("\nSaved calibration_results.json")

# Calibration curve plot for best model
oof = np.load(PRED / "oof_proba_finbert_fintwitter_10ep_fixtext.npy")
max_proba = oof.max(axis=1)
correct = (oof.argmax(axis=1) == y)
bins = np.linspace(0.33, 1.0, 11)
mids, accs, cnts = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (max_proba >= lo) & (max_proba < hi)
    if m.sum() >= 20:
        mids.append((lo + hi) / 2)
        accs.append(correct[m].mean())
        cnts.append(int(m.sum()))
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot([0.33, 1], [0.33, 1], "k--", alpha=0.5, label="Perfect calibration")
ax.plot(mids, accs, "o-", color="#1f77b4", label="FinBERT 10ep fix_text")
ax.set_xlabel("Predicted confidence")
ax.set_ylabel("Observed accuracy")
ax.set_title("Calibration curve (OOF, 10-fold)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "calibration_curve.pdf", dpi=300, bbox_inches="tight")
plt.savefig(FIG / "calibration_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved calibration_curve.{pdf,png}")

# ============================================================
# 4.7  Statistical significance: Wilcoxon + bootstrap CI
# ============================================================
print()
print("=" * 60)
print("4.7  Statistical significance")
print("=" * 60)

from scipy.stats import wilcoxon
from sklearn.utils import resample

best_folds = json.loads((TAB / "finbert_fintwitter_10ep_fixtext_result.json").read_text())["per_fold_f1"]
second_folds = json.loads((TAB / "finbert_fintwitter_7ep_result.json").read_text())["per_fold_f1"]

stat, p = wilcoxon(best_folds, second_folds, alternative="greater")
print(f"Wilcoxon (FinBERT-10ep+fix > FinBERT-7ep): statistic={stat:.3f}, p={p:.4f}")

# Bootstrap CI: best individual vs second-best (paired, per-sample)
oof_best = np.load(PRED / "oof_proba_finbert_fintwitter_10ep_fixtext.npy")
oof_second = np.load(PRED / "oof_proba_finbert_fintwitter_7ep.npy")

rng = np.random.RandomState(42)
n = len(y)
f1_diffs = []
for _ in range(1000):
    idx = rng.randint(0, n, n)
    f1_a = f1_score(y[idx], oof_best[idx].argmax(1), average="macro")
    f1_b = f1_score(y[idx], oof_second[idx].argmax(1), average="macro")
    f1_diffs.append(f1_a - f1_b)
ci_low, ci_high = np.percentile(f1_diffs, [2.5, 97.5])
print(f"Bootstrap 95% CI (best - second): [{ci_low:.4f}, {ci_high:.4f}]  "
      f"{'(inclui 0 - nao significativo)' if ci_low <= 0 else '(diferenca real)'}")

# Bootstrap: ensemble vs best individual
with open(TAB / "ensemble_optimal_result.json") as f:
    opt = json.load(f)
WEIGHTS = opt["models"]
ens_sum = np.zeros((n, 3))
wsum = 0.0
for tag, w in WEIGHTS.items():
    p_oof = PRED / f"oof_proba_{tag}.npy"
    if p_oof.exists():
        ens_sum += np.load(p_oof) * w
        wsum += w
oof_ens = ens_sum / wsum

f1_diffs_ens = []
for _ in range(1000):
    idx = rng.randint(0, n, n)
    f1_a = f1_score(y[idx], oof_ens[idx].argmax(1), average="macro")
    f1_b = f1_score(y[idx], oof_best[idx].argmax(1), average="macro")
    f1_diffs_ens.append(f1_a - f1_b)
ci_low_e, ci_high_e = np.percentile(f1_diffs_ens, [2.5, 97.5])
print(f"Bootstrap 95% CI (ensemble - best individual): [{ci_low_e:.4f}, {ci_high_e:.4f}]  "
      f"{'(inclui 0)' if ci_low_e <= 0 else '(diferenca real)'}")

sig = {
    "wilcoxon_best_vs_second": {"statistic": float(stat), "p_value": float(p)},
    "bootstrap_best_vs_second_ci95": [float(ci_low), float(ci_high)],
    "bootstrap_ensemble_vs_best_ci95": [float(ci_low_e), float(ci_high_e)],
    "ensemble_oof_f1": float(f1_score(y, oof_ens.argmax(1), average="macro")),
    "best_oof_f1": float(f1_score(y, oof_best.argmax(1), average="macro")),
    "n_bootstrap": 1000,
}
(TAB / "statistical_significance.json").write_text(json.dumps(sig, indent=2))
print("Saved statistical_significance.json")

# Checkpoint
cp = BASE / "results" / "progress_checkpoint.json"
state = json.loads(cp.read_text()) if cp.exists() else {}
state["phase_2_analysis"] = {"done": True, "timestamp": time.time()}
cp.write_text(json.dumps(state, indent=2))
print("\nCheckpoint phase_2_analysis guardado.")
