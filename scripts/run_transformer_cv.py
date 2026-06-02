"""Run 5-fold transformer fine-tuning for one Hugging Face backbone.

This script mirrors the final notebook protocol: raw tweet text, stratified
folds, class-weighted cross entropy, AdamW, warmup, max length 96 by default,
and averaged test probabilities across folds.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


SEED = 42


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--maxlen", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args()


def maybe_freeze_for_cpu(model: torch.nn.Module, top_layers: int = 4) -> None:
    """CPU fallback: train only the classifier and final encoder layers."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    num_layers = getattr(model.config, "num_hidden_layers", None)
    unfreeze_from = max(0, num_layers - top_layers) if num_layers is not None else None
    for name, parameter in model.named_parameters():
        if "classifier" in name or "score" in name:
            parameter.requires_grad = True
            continue
        if unfreeze_from is not None and "encoder.layer." in name:
            try:
                layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            if layer_idx >= unfreeze_from:
                parameter.requires_grad = True


@torch.no_grad()
def predict_proba(model, tokenizer, texts, device, maxlen, eval_batch_size, amp_dtype):
    model.eval()
    outputs = []
    for start in range(0, len(texts), eval_batch_size):
        batch = texts[start : start + eval_batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=maxlen,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(**encoded).logits
        else:
            logits = model(**encoded).logits
        outputs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.vstack(outputs)


def main() -> None:
    args = parse_args()
    seed_all()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    table_dir = out_dir / "tables"
    pred_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)
    train_texts = train["text"].astype(str).tolist()
    test_texts = test["text"].astype(str).tolist()
    y = train["label"].to_numpy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        batch_size = args.batch_size or 16
        amp_dtype = torch.bfloat16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"VRAM {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        batch_size = args.batch_size or 8
        amp_dtype = None
        torch.set_num_threads(min(12, os.cpu_count() or 1))
        print("CPU fallback: classifier + top encoder layers only")

    counts = np.bincount(y, minlength=3)
    class_weights = torch.tensor(len(y) / (3 * counts), dtype=torch.float32, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_prob_sum = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []
    started = time.time()

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_texts, y), start=1):
        seed_all(SEED + fold)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=3,
            ignore_mismatched_sizes=True,
        )
        if device == "cpu":
            maybe_freeze_for_cpu(model)
        model.to(device)
        model.train()

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = int(np.ceil(len(train_idx) / batch_size))
        total_steps = steps_per_epoch * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        fold_texts = [train_texts[i] for i in train_idx]
        fold_labels = y[train_idx]
        for epoch in range(args.epochs):
            order = np.random.permutation(len(fold_texts))
            running = 0.0
            epoch_started = time.time()
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                batch_texts = [fold_texts[i] for i in idx]
                batch_labels = torch.tensor(fold_labels[idx], dtype=torch.long, device=device)
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=args.maxlen,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                optimizer.zero_grad(set_to_none=True)
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        loss = loss_fn(model(**encoded).logits, batch_labels)
                else:
                    loss = loss_fn(model(**encoded).logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                running += float(loss.item())
            print(
                f"fold {fold}/{args.n_folds} epoch {epoch + 1}/{args.epochs} "
                f"loss={running / steps_per_epoch:.4f} "
                f"time={time.time() - epoch_started:.0f}s"
            )

        valid_texts = [train_texts[i] for i in valid_idx]
        valid_proba = predict_proba(
            model, tokenizer, valid_texts, device, args.maxlen, args.eval_batch_size, amp_dtype
        )
        oof[valid_idx] = valid_proba
        valid_pred = valid_proba.argmax(axis=1)
        fold_f1 = f1_score(y[valid_idx], valid_pred, average="macro")
        fold_scores.append(fold_f1)
        print(f"fold {fold} valid macro-F1={fold_f1:.6f}")

        test_prob_sum += predict_proba(
            model, tokenizer, test_texts, device, args.maxlen, args.eval_batch_size, amp_dtype
        )

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    test_proba = test_prob_sum / args.n_folds
    oof_pred = oof.argmax(axis=1)
    test_pred = test_proba.argmax(axis=1)

    result = {
        "tag": args.tag,
        "model_id": args.model_name,
        "F1-macro": float(f1_score(y, oof_pred, average="macro")),
        "F1-macro_std": float(np.std(fold_scores)),
        "F1-weighted": float(f1_score(y, oof_pred, average="weighted")),
        "Accuracy": float(accuracy_score(y, oof_pred)),
        "Precision-macro": float(precision_score(y, oof_pred, average="macro", zero_division=0)),
        "Recall-macro": float(recall_score(y, oof_pred, average="macro", zero_division=0)),
        "per_fold_f1": [float(score) for score in fold_scores],
        "elapsed_min": float((time.time() - started) / 60),
        "test_dist": np.bincount(test_pred, minlength=3).astype(int).tolist(),
        "recipe": {
            "epochs": args.epochs,
            "maxlen": args.maxlen,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": batch_size,
            "eval_batch_size": args.eval_batch_size,
            "n_folds": args.n_folds,
            "class_weighted_loss": True,
            "raw_text": True,
        },
    }

    np.save(pred_dir / f"oof_proba_{args.tag}.npy", oof)
    np.save(pred_dir / f"test_proba_{args.tag}.npy", test_proba.astype(np.float32))
    pd.DataFrame({"id": test["id"], "label": test_pred}).to_csv(
        pred_dir / f"pred_{args.tag}.csv", index=False
    )
    (table_dir / f"{args.tag}_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
