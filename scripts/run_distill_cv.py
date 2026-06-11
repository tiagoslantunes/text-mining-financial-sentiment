"""Knowledge distillation: compress the optimal ensemble into ONE FinBERT.

Teacher = weighted soft-vote ensemble (results/tables/ensemble_optimal_result.json).
Student = nickmuchi/finbert-tone-finetuned-fintwitter-classification.

Loss = (1-alpha) * CE(hard labels, class-weighted)
     + alpha * T^2 * KL(student_T || teacher_T)        (Hinton et al., 2015)

Teacher soft targets are the leak-free OOF probabilities, so each training sample's
target comes from folds that never saw that sample. The distilled student is a single
classification model -> rule-compliant for tm_final_33 (Guidelines section 5.2).

Run: python scripts/run_distill_cv.py --tag finbert_distilled --epochs 10 --n-folds 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import tempfile
import time
from pathlib import Path

import re

import ftfy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

SEED = 42

_TRUNC_RE = re.compile(r'[�…°]+\s*(https?://\S*)?$')
_URL_RE = re.compile(r'https?://\S+')
_TRAIL_RE = re.compile(r'[\s\-–:]+$')


def fix_tweet(text: str) -> str:
    text = ftfy.fix_text(str(text))
    text = _TRUNC_RE.sub('', text)
    text = _URL_RE.sub('', text)
    return _TRAIL_RE.sub('', text).strip()


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="nickmuchi/finbert-tone-finetuned-fintwitter-classification")
    p.add_argument("--tag", default="finbert_distilled")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--maxlen", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5, help="weight of the KD term")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--max-folds", type=int, default=0,
                   help="if >0, train only the first k folds (fast hyperparameter proxy)")
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--weights-json", default="results/tables/ensemble_optimal_result.json")
    p.add_argument("--train-csv", default="data/raw/train.csv")
    p.add_argument("--test-csv", default="data/raw/test.csv")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


@torch.no_grad()
def predict_proba(model, tokenizer, texts, device, maxlen, bs, amp):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tokenizer(texts[i:i + bs], padding=True, truncation=True,
                        max_length=maxlen, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp):
                logits = model(**enc).logits
        else:
            logits = model(**enc).logits
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.vstack(out)


def main() -> None:
    args = parse_args()
    seed_all()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    table_dir = out_dir / "tables"

    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)
    texts = [fix_tweet(t) for t in train["text"].astype(str)]
    test_texts = [fix_tweet(t) for t in test["text"].astype(str)]
    y = train["label"].to_numpy()

    # ---- teacher soft targets: weighted ensemble of OOF probabilities ----
    opt = json.loads(Path(args.weights_json).read_text())
    weights = opt["models"]
    t_sum = np.zeros((len(train), 3), dtype=np.float64)
    t_w = 0.0
    used = []
    for tag, w in weights.items():
        p = pred_dir / f"oof_proba_{tag}.npy"
        if p.exists():
            t_sum += np.load(p) * w
            t_w += w
            used.append(tag)
    teacher = (t_sum / t_w).astype(np.float32)
    teacher_f1 = f1_score(y, teacher.argmax(1), average="macro")
    print(f"Teacher: {len(used)} models, OOF F1-macro={teacher_f1:.4f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.float16
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    counts = np.bincount(y, minlength=3)
    class_weights = torch.tensor(len(y) / (3 * counts), dtype=torch.float32, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)

    T, alpha = args.temperature, args.alpha
    teacher_t = torch.tensor(teacher)
    # soften teacher with temperature (on log-probs)
    teacher_soft_all = torch.softmax(torch.log(teacher_t + 1e-9) / T, dim=1)

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_sum = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []
    started = time.time()

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(texts, y), start=1):
        seed_all(SEED + fold)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=3, ignore_mismatched_sizes=True,
            dtype=torch.float32).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        steps_per_epoch = int(np.ceil(len(tr_idx) / args.batch_size))
        total_steps = steps_per_epoch * args.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, int(args.warmup_ratio * total_steps), total_steps)
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

        ft = [texts[i] for i in tr_idx]
        fl = y[tr_idx]
        fsoft = teacher_soft_all[tr_idx]

        best_f1, best_va = -1.0, None
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "best.pt"
            for epoch in range(args.epochs):
                model.train()
                order = np.random.permutation(len(ft))
                running = 0.0
                t0 = time.time()
                for start in range(0, len(order), args.batch_size):
                    idx = order[start:start + args.batch_size]
                    bt = [ft[i] for i in idx]
                    bl = torch.tensor(fl[idx], dtype=torch.long, device=device)
                    bsoft = fsoft[idx].to(device)
                    enc = tokenizer(bt, padding=True, truncation=True,
                                    max_length=args.maxlen, return_tensors="pt")
                    enc = {k: v.to(device) for k, v in enc.items()}
                    optimizer.zero_grad(set_to_none=True)
                    if device == "cuda":
                        with torch.autocast(device_type="cuda", dtype=amp):
                            logits = model(**enc).logits
                            loss_ce = ce_loss(logits, bl)
                            log_student_T = F.log_softmax(logits / T, dim=1)
                            loss_kd = F.kl_div(log_student_T, bsoft, reduction="batchmean") * (T * T)
                            loss = (1 - alpha) * loss_ce + alpha * loss_kd
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logits = model(**enc).logits
                        loss = (1 - alpha) * ce_loss(logits, bl) + alpha * F.kl_div(
                            F.log_softmax(logits / T, dim=1), bsoft,
                            reduction="batchmean") * (T * T)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    scheduler.step()
                    running += float(loss.item())

                va_texts = [texts[i] for i in va_idx]
                va_proba = predict_proba(model, tokenizer, va_texts, device,
                                         args.maxlen, args.eval_batch_size, amp)
                ep_f1 = f1_score(y[va_idx], va_proba.argmax(1), average="macro")
                star = " *** best ***" if ep_f1 > best_f1 else ""
                print(f"fold {fold}/{args.n_folds} epoch {epoch+1}/{args.epochs} "
                      f"loss={running/steps_per_epoch:.4f} val_f1={ep_f1:.6f} "
                      f"time={time.time()-t0:.0f}s{star}", flush=True)
                if ep_f1 > best_f1:
                    best_f1 = ep_f1
                    best_va = va_proba.copy()
                    torch.save(model.state_dict(), ckpt)

            model.load_state_dict(torch.load(ckpt, map_location=device))
            test_proba = predict_proba(model, tokenizer, test_texts, device,
                                       args.maxlen, args.eval_batch_size, amp)

        oof[va_idx] = best_va
        test_sum += test_proba
        fold_scores.append(best_f1)
        print(f"fold {fold} best val macro-F1={best_f1:.6f}", flush=True)
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        if args.max_folds and fold >= args.max_folds:
            print(f"PROXY RESULT tag={args.tag} mean_fold_f1={np.mean(fold_scores):.6f} "
                  f"folds={[round(s,4) for s in fold_scores]}", flush=True)
            return

    test_proba = test_sum / args.n_folds
    oof_pred = oof.argmax(1)

    result = {
        "tag": args.tag,
        "model_id": args.model_name,
        "F1-macro": float(f1_score(y, oof_pred, average="macro")),
        "F1-macro_std": float(np.std(fold_scores)),
        "F1-weighted": float(f1_score(y, oof_pred, average="weighted")),
        "Accuracy": float(accuracy_score(y, oof_pred)),
        "Precision-macro": float(precision_score(y, oof_pred, average="macro", zero_division=0)),
        "Recall-macro": float(recall_score(y, oof_pred, average="macro", zero_division=0)),
        "per_fold_f1": [float(s) for s in fold_scores],
        "teacher_oof_f1": float(teacher_f1),
        "teacher_models": used,
        "elapsed_min": float((time.time() - started) / 60),
        "test_dist": np.bincount(test_proba.argmax(1), minlength=3).astype(int).tolist(),
        "recipe": {
            "epochs": args.epochs, "maxlen": args.maxlen, "lr": args.lr,
            "batch_size": args.batch_size, "n_folds": args.n_folds,
            "alpha": alpha, "temperature": T, "fix_text": True,
            "loss": "(1-a)*weighted_CE + a*T^2*KL(student||teacher)",
        },
    }

    np.save(pred_dir / f"oof_proba_{args.tag}.npy", oof)
    np.save(pred_dir / f"test_proba_{args.tag}.npy", test_proba.astype(np.float32))
    pd.DataFrame(test_proba, columns=["p0", "p1", "p2"]).to_csv(
        pred_dir / f"prob_test_{args.tag}.csv", index=False)
    pd.DataFrame({"id": test["id"], "label": test_proba.argmax(1)}).to_csv(
        pred_dir / f"pred_{args.tag}.csv", index=False)
    (table_dir / f"{args.tag}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "recipe"}, indent=2))


if __name__ == "__main__":
    main()
