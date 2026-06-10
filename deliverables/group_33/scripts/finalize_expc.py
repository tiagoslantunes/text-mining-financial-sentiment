"""Post-process EXP C (debertav3_large_6ep_fixtext) and update ensemble.

Run after EXP C finishes:
  python scripts/finalize_expc.py

Actions:
1. Convert .npy → prob_test_debertav3_large_6ep_fixtext.csv
2. Recompute ensemble (weighted soft-vote) with all valid models
3. Save updated pred_33.csv if ensemble improves
4. Print summary of all model F1s and ensemble result
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

BASE   = Path(__file__).resolve().parent.parent
PRED   = BASE / "results" / "predictions"
TABLE  = BASE / "results" / "tables"

# ── 1. Convert new .npy → CSV ────────────────────────────────────────────────
npy_path = PRED / "test_proba_debertav3_large_6ep_fixtext.npy"
csv_path = PRED / "prob_test_debertav3_large_6ep_fixtext.csv"

if not npy_path.exists():
    raise FileNotFoundError(f"EXP C .npy not found: {npy_path}")

proba = np.load(npy_path)
pd.DataFrame(proba, columns=["p0", "p1", "p2"]).to_csv(csv_path, index=False)
print(f"Saved {csv_path.name}  shape={proba.shape}")

# ── 2. Load OOF probabilities and labels ─────────────────────────────────────
train = pd.read_csv(BASE / "data" / "raw" / "train.csv")
test  = pd.read_csv(BASE / "data" / "raw" / "test.csv")
y     = train["label"].to_numpy()
N_TEST = len(test)

# ── 3. Load all OOF proba files for ensemble F1 estimation ───────────────────
# (We use OOF to measure ensemble quality on training data)
ENSEMBLE_MODELS = {
    "finbert_fintwitter_10ep_fixtext": 1.0,   # primary in notebook
    "finbert_fintwitter_10ep":         1.0,
    "finbert_fintwitter_7ep":          1.0,
    "finbert_fintwitter":              1.0,
    "roberta_large_ts_v2":             1.0,
    "debertav3_large_6ep":             1.0,
    "deberta_base_finance_fixtext":    0.5,
}
CANDIDATE_TAG = "debertav3_large_6ep_fixtext"

def load_oof(tag):
    p = PRED / f"oof_proba_{tag}.npy"
    if p.exists():
        return np.load(p)
    return None

def ensemble_f1(model_dict):
    oof_sum   = np.zeros((len(y), 3), dtype=np.float64)
    total_w   = 0.0
    loaded    = []
    for tag, w in model_dict.items():
        arr = load_oof(tag)
        if arr is None:
            continue
        oof_sum += arr * w
        total_w += w
        loaded.append(tag)
    if total_w == 0:
        return 0.0, []
    preds = (oof_sum / total_w).argmax(axis=1)
    return f1_score(y, preds, average="macro"), loaded

# Baseline ensemble (without EXP C)
base_f1, base_loaded = ensemble_f1(ENSEMBLE_MODELS)
print(f"\nBaseline ensemble ({len(base_loaded)} models): OOF F1 = {base_f1:.4f}")

# + EXP C
with_expc = dict(ENSEMBLE_MODELS)
with_expc[CANDIDATE_TAG] = 1.0
expc_f1, expc_loaded = ensemble_f1(with_expc)
print(f"+ {CANDIDATE_TAG} (w=1.0):            OOF F1 = {expc_f1:.4f}  ({expc_f1 - base_f1:+.4f})")

# + EXP C half weight
with_expc_half = dict(ENSEMBLE_MODELS)
with_expc_half[CANDIDATE_TAG] = 0.5
expc_half_f1, _ = ensemble_f1(with_expc_half)
print(f"+ {CANDIDATE_TAG} (w=0.5):            OOF F1 = {expc_half_f1:.4f}  ({expc_half_f1 - base_f1:+.4f})")

# ── 4. Choose best configuration ─────────────────────────────────────────────
best_f1 = base_f1
best_models = ENSEMBLE_MODELS
best_label  = "baseline (no EXP C)"

if expc_f1 > best_f1 + 0.0001:
    best_f1     = expc_f1
    best_models = with_expc
    best_label  = f"+ EXP C w=1.0"

if expc_half_f1 > best_f1 + 0.0001:
    best_f1     = expc_half_f1
    best_models = with_expc_half
    best_label  = f"+ EXP C w=0.5"

print(f"\nBest config: {best_label}  →  OOF F1 = {best_f1:.4f}")

# ── 5. Build test-set ensemble with best config ───────────────────────────────
test_sum   = np.zeros((N_TEST, 3), dtype=np.float64)
total_w    = 0.0
for tag, w in best_models.items():
    p = PRED / f"prob_test_{tag}.csv"
    if p.exists():
        arr = pd.read_csv(p)[["p0","p1","p2"]].values.astype(np.float64)
        if arr.shape == (N_TEST, 3):
            test_sum += arr * w
            total_w  += w

test_proba = test_sum / total_w
test_pred  = test_proba.argmax(axis=1)

# ── 6. Save pred_33.csv ───────────────────────────────────────────────────────
pred_out = BASE / "pred_33.csv"
pd.DataFrame({"id": test["id"], "label": test_pred}).to_csv(pred_out, index=False)
print(f"\nSaved {pred_out}  dist={np.bincount(test_pred.astype(int), minlength=3).tolist()}")

# ── 7. Save updated ensemble result JSON ─────────────────────────────────────
result = {
    "ensemble_label": best_label,
    "oof_f1_macro":   float(best_f1),
    "models": {t: float(w) for t, w in best_models.items()},
    "test_dist": np.bincount(test_pred.astype(int), minlength=3).astype(int).tolist(),
}
out_json = TABLE / "ensemble_final_result.json"
out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"Saved {out_json.name}")

# ── 8. Individual model summary ───────────────────────────────────────────────
print("\n── Individual model F1 (OOF) ────────────────────────────────")
all_tags = list(ENSEMBLE_MODELS.keys()) + [CANDIDATE_TAG]
for tag in all_tags:
    jf = TABLE / f"{tag}_result.json"
    if jf.exists():
        d = json.loads(jf.read_text(encoding="utf-8"))
        print(f"  {tag:<45} {d['F1-macro']:.4f}")
