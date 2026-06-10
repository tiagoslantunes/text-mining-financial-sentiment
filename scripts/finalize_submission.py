# -*- coding: utf-8 -*-
"""Final submission assembly after distillation.

1. pred_33.csv <- distilled single model (rule-compliant: Guidelines 5.2)
2. Confusion-matrix figure for the submitted model (normalised + absolute)
3. Updated results bar chart for the report
4. Full assertions + checkpoint

Run: python scripts/finalize_submission.py
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

BASE = Path(__file__).resolve().parent.parent
PRED = BASE / "results" / "predictions"
TAB = BASE / "results" / "tables"
FIG = BASE / "results" / "figures"

TAG = "finbert_distilled"
CLASSES = ["Bearish", "Bullish", "Neutral"]

train = pd.read_csv(BASE / "data" / "raw" / "train.csv")
test = pd.read_csv(BASE / "data" / "raw" / "test.csv")
y = train["label"].to_numpy()

# ---- 1. pred_33.csv from the distilled single model -------------------------
test_proba = np.load(PRED / f"test_proba_{TAG}.npy")
test_pred = test_proba.argmax(1)
sub = pd.DataFrame({"id": test["id"], "label": test_pred.astype(int)})
sub.to_csv(BASE / "pred_33.csv", index=False)
sub.to_csv(PRED / "pred_final_distilled.csv", index=False)

assert len(sub) == 2388
assert list(sub.columns) == ["id", "label"]
assert set(sub["label"].unique()) == {0, 1, 2}
assert sub["label"].isna().sum() == 0
assert sub["id"].is_unique
assert sub["label"].value_counts(normalize=True).min() > 0.01
print(f"pred_33.csv gerado pelo modelo destilado ({TAG})")
print(sub["label"].value_counts().sort_index().rename({0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}).to_string())

# ---- 2. Confusion matrix figure ---------------------------------------------
oof = np.load(PRED / f"oof_proba_{TAG}.npy")
oof_pred = oof.argmax(1)
res = json.loads((TAB / f"{TAG}_result.json").read_text())
f1m = res["F1-macro"]

cm = confusion_matrix(y, oof_pred)
cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, mat, title, fmt in [
        (axes[0], cmn, "Normalised (recall per class)", ".2f"),
        (axes[1], cm, "Absolute counts", "d")]:
    im = ax.imshow(mat, cmap="Greens", aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    thresh = mat.max() / 2
    for i in range(3):
        for j in range(3):
            v = mat[i, j]
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=11,
                    color="white" if v > thresh else "black")
fig.suptitle(f"Submitted model (distilled FinBERT) - 10-fold OOF, macro-F1 = {f1m:.4f}",
             fontsize=11)
plt.tight_layout()
plt.savefig(FIG / "confusion_matrix_distilled.pdf", dpi=300, bbox_inches="tight")
plt.savefig(FIG / "confusion_matrix_distilled.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved confusion_matrix_distilled.{pdf,png}")

# ---- 3. Results bar chart (top configurations) ------------------------------
opt = json.loads((TAB / "ensemble_optimal_result.json").read_text())
rows = [("Ensemble 8+ models (extra, teacher)", opt["oof_f1_macro"], "#2ca02c")]
rows.append((f"Distilled FinBERT (SUBMITTED)", f1m, "#649b00"))

chart_tags = [
    ("finbert_fintwitter_10ep_fixtext", "FinBERT 10ep + fix_text"),
    ("finbert_fintwitter_llrd_7ep", "FinBERT 7ep + LLRD"),
    ("finbert_fintwitter_7ep", "FinBERT 7ep"),
    ("roberta_large_ts_v2", "Twitter-RoBERTa-large"),
    ("debertav3_large_8ep_fixtext", "DeBERTa-v3-large 8ep + fix_text"),
    ("debertav3_large_6ep_fixtext", "DeBERTa-v3-large 6ep + fix_text"),
    ("deberta_base_finance_fixtext", "DeBERTa-v3-base finance"),
    ("gpt2_decoder", "GPT-2 decoder (extra)"),
]
for tag, label in chart_tags:
    p = TAB / f"{tag}_result.json"
    if p.exists():
        d = json.loads(p.read_text())
        rows.append((label, d["F1-macro"], "#1f77b4"))

# classical baselines for context
rows.append(("LightGBM Optuna + RoBERTa (classical best)", 0.8021, "#999999"))
rows.append(("Stacking 6 learners (extra)", 0.8483, "#999999"))

rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.2))
names = [r[0] for r in rows]
vals = [r[1] for r in rows]
cols = [r[2] for r in rows]
bars = ax.barh(names, vals, color=cols, alpha=0.9)
for bar, v in zip(bars, vals):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}", va="center", fontsize=8)
ax.set_xlabel("OOF F1-macro (10-fold stratified CV, seed=42)")
ax.set_xlim(0.75, max(vals) + 0.03)
ax.set_title("Model comparison - identical OOF protocol")
plt.tight_layout()
plt.savefig(FIG / "report_results.png", dpi=150, bbox_inches="tight")
plt.savefig(FIG / "results_final.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("Saved report_results.png / results_final.pdf")

# ---- 4. Metrics summary for the report --------------------------------------
summary = {
    "submitted_tag": TAG,
    "submitted_oof_f1": f1m,
    "submitted_acc": res["Accuracy"],
    "submitted_prec": res["Precision-macro"],
    "submitted_rec": res["Recall-macro"],
    "teacher_f1": res.get("teacher_oof_f1"),
    "per_class_f1": {c: float(f1_score(y == i, oof_pred == i)) for i, c in enumerate(CLASSES)},
    "test_dist": np.bincount(test_pred, minlength=3).astype(int).tolist(),
}
(TAB / "submission_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))

cp = BASE / "results" / "progress_checkpoint.json"
state = json.loads(cp.read_text()) if cp.exists() else {}
state["phase_final_submission"] = {"done": True, "timestamp": time.time(), "oof_f1": f1m}
cp.write_text(json.dumps(state, indent=2))
print("Checkpoint guardado.")
