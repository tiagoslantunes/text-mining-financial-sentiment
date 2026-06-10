# -*- coding: utf-8 -*-
"""Re-optimise ensemble weights over all available OOF probability files.

Method: coordinate ascent on the weight grid {0, 0.25, 0.5, 0.75, 1.0},
initialised from the current ensemble_optimal_result.json, with new models
starting at weight 0. Repeats passes until no single-weight change improves
OOF F1-macro. Deterministic and exhaustive per-coordinate.

Only overwrites ensemble_optimal_result.json when the result improves.
Run: python scripts/reoptimize_ensemble.py
"""

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

BASE = Path(__file__).resolve().parent.parent
PRED = BASE / "results" / "predictions"
TAB = BASE / "results" / "tables"

train = pd.read_csv(BASE / "data" / "raw" / "train.csv")
y = train["label"].to_numpy()

CANDIDATES = [
    "finbert_fintwitter_10ep_fixtext",
    "finbert_fintwitter_10ep",
    "finbert_fintwitter_7ep",
    "finbert_fintwitter",
    "finbert_fintwitter_llrd_7ep",
    "roberta_large_ts_v2",
    "debertav3_large_6ep",
    "debertav3_large_6ep_fixtext",
    "debertav3_large_8ep_fixtext",
    "deberta_base_finance_fixtext",
]

oofs = {}
for tag in CANDIDATES:
    p = PRED / f"oof_proba_{tag}.npy"
    if p.exists():
        oofs[tag] = np.load(p).astype(np.float64)
print(f"Modelos disponiveis: {len(oofs)}")
for t in oofs:
    f1_ind = f1_score(y, oofs[t].argmax(1), average="macro")
    print(f"  {t:<40} individual OOF F1 = {f1_ind:.4f}")

GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def ens_f1(weights: dict) -> float:
    s = np.zeros((len(y), 3))
    tw = 0.0
    for t, w in weights.items():
        if w > 0 and t in oofs:
            s += oofs[t] * w
            tw += w
    if tw == 0:
        return 0.0
    return f1_score(y, (s / tw).argmax(1), average="macro")


# initialise from current optimum; new models start at 0
cur = json.loads((TAB / "ensemble_optimal_result.json").read_text())
weights = {t: 0.0 for t in oofs}
weights.update({t: w for t, w in cur["models"].items() if t in oofs})
best_f1 = ens_f1(weights)
print(f"\nPonto de partida: {best_f1:.4f}  (config actual)")

improved = True
passes = 0
while improved and passes < 10:
    improved = False
    passes += 1
    for tag in oofs:
        best_w = weights[tag]
        for w in GRID:
            if w == weights[tag]:
                continue
            trial = dict(weights)
            trial[tag] = w
            f1 = ens_f1(trial)
            if f1 > best_f1 + 1e-6:
                best_f1 = f1
                best_w = w
        if best_w != weights[tag]:
            weights[tag] = best_w
            improved = True
            print(f"  pass {passes}: {tag} -> w={best_w}  (F1={best_f1:.4f})")

final = {t: w for t, w in weights.items() if w > 0}
print(f"\nMelhor configuracao ({len(final)} modelos): OOF F1-macro = {best_f1:.4f}")
for t, w in sorted(final.items(), key=lambda x: -x[1]):
    print(f"  w={w:<5} {t}")

if best_f1 > cur["oof_f1_macro"] + 1e-6:
    out = {
        "oof_f1_macro": float(best_f1),
        "n_folds": 10,
        "optimisation": "coordinate ascent, grid {0,0.25,0.5,0.75,1.0}",
        "models": final,
    }
    (TAB / "ensemble_optimal_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nensemble_optimal_result.json ACTUALIZADO ({cur['oof_f1_macro']:.4f} -> {best_f1:.4f})")
else:
    print(f"\nSem melhoria vs {cur['oof_f1_macro']:.4f} - JSON mantido.")
