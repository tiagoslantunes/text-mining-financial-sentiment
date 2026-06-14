# -*- coding: utf-8 -*-
"""Fine-tune GPT-2 (a decoder LM) for sequence classification — extra work,
"Classification Models" decoder criterion. Reproduces gpt2_decoder_result.json
(OOF F1-macro 0.7724).

GPT2ForSequenceClassification pools the last non-pad token's hidden state into a
3-class head. Trained end-to-end under the shared 5-fold stratified protocol
(seed 42). This demonstrates that a *decoder* (causal) model can be fine-tuned
for classification, in contrast to the bidirectional encoders used elsewhere.

Run from the project root (GPU recommended):
    python scripts/run_gpt2_decoder.py --tag gpt2_decoder --epochs 4 --n-folds 5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, get_linear_schedule_with_warmup

BASE = Path(__file__).resolve().parent.parent
SEED = 42


def seed_all(s=SEED):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gpt2_decoder")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()
    seed_all()

    train = pd.read_csv(BASE / "data" / "raw" / "train.csv")
    texts = train["text"].astype(str).tolist()
    y = train["label"].to_numpy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token   # GPT-2 has no pad token by default

    counts = np.bincount(y, minlength=3)
    cw = torch.tensor(len(y) / (3 * counts), dtype=torch.float32, device=device)
    splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), 3), dtype=np.float32)
    fold_f1 = []
    t0 = time.time()

    @torch.no_grad()
    def predict(model, idx):
        model.eval()
        out = []
        for s in range(0, len(idx), 32):
            b = [texts[i] for i in idx[s:s+32]]
            enc = tok(b, padding=True, truncation=True, max_length=args.maxlen, return_tensors="pt").to(device)
            out.append(torch.softmax(model(**enc).logits.float(), 1).cpu().numpy())
        return np.vstack(out)

    for fold, (tr, va) in enumerate(splitter.split(texts, y), 1):
        seed_all(SEED + fold)
        model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)
        model.config.pad_token_id = tok.pad_token_id
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        steps = int(np.ceil(len(tr) / args.batch_size)) * args.epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        loss_fn = nn.CrossEntropyLoss(weight=cw)

        for ep in range(args.epochs):
            model.train()
            order = np.random.permutation(len(tr))
            for s in range(0, len(order), args.batch_size):
                bi = tr[order[s:s+args.batch_size]]
                bt = [texts[i] for i in bi]
                bl = torch.tensor(y[bi], dtype=torch.long, device=device)
                enc = tok(bt, padding=True, truncation=True, max_length=args.maxlen, return_tensors="pt").to(device)
                opt.zero_grad()
                loss = loss_fn(model(**enc).logits, bl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
        vp = predict(model, va)
        oof[va] = vp
        f = f1_score(y[va], vp.argmax(1), average="macro")
        fold_f1.append(f)
        print(f"fold {fold}/{args.n_folds} macro-F1={f:.4f}")
        del model; torch.cuda.empty_cache() if device == "cuda" else None

    pred = oof.argmax(1)
    res = {
        "Model": "GPT-2 decoder fine-tuned (GPT2ForSequenceClassification)",
        "F1-macro": float(f1_score(y, pred, average="macro")),
        "F1-macro_std": float(np.std(fold_f1)),
        "Accuracy": float(accuracy_score(y, pred)),
        "Precision-macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "Recall-macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "per_fold_f1": [float(x) for x in fold_f1],
        "elapsed_min": (time.time() - t0) / 60,
    }
    (BASE / "results" / "tables" / f"{args.tag}_result.json").write_text(json.dumps(res, indent=2))
    np.save(BASE / "results" / "predictions" / f"oof_proba_{args.tag}.npy", oof)
    print(json.dumps({k: v for k, v in res.items() if k != "per_fold_f1"}, indent=2))


if __name__ == "__main__":
    main()
