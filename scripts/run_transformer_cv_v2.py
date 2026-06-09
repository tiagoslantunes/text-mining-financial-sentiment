"""Enhanced 5-fold transformer fine-tuning script.

Key improvements over v1:
- Best-checkpoint saving per fold (val F1, not last epoch)
- Label smoothing
- Cosine warmup schedule
- Layer-wise learning rate decay (LLRD)
- Gradient accumulation
- fp16 instead of bf16 (avoids NaN with DeBERTa)
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--llrd", type=float, default=0.0, help="Layer-wise LR decay factor (0=disabled, 0.9=mild)")
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16", "none"], default="fp16")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args()


def build_optimizer_with_llrd(model, base_lr: float, weight_decay: float, llrd: float):
    """Group parameters by layer with exponentially decaying LR."""
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if llrd == 0.0 or num_layers is None:
        decay = [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)]
        no_decay_p = [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)]
        groups = [
            {"params": decay, "lr": base_lr, "weight_decay": weight_decay},
            {"params": no_decay_p, "lr": base_lr, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=base_lr)

    groups = []
    for layer_idx in range(num_layers):
        layer_lr = base_lr * (llrd ** (num_layers - 1 - layer_idx))
        layer_prefix = f"encoder.layer.{layer_idx}."
        decay = [
            p for n, p in model.named_parameters()
            if p.requires_grad and layer_prefix in n and not any(nd in n for nd in no_decay)
        ]
        no_decay_p = [
            p for n, p in model.named_parameters()
            if p.requires_grad and layer_prefix in n and any(nd in n for nd in no_decay)
        ]
        if decay:
            groups.append({"params": decay, "lr": layer_lr, "weight_decay": weight_decay})
        if no_decay_p:
            groups.append({"params": no_decay_p, "lr": layer_lr, "weight_decay": 0.0})

    # embeddings + classifier at base_lr
    other = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not any(f"encoder.layer.{i}." in n for i in range(num_layers))
    ]
    if other:
        groups.append({"params": other, "lr": base_lr, "weight_decay": weight_decay})

    return torch.optim.AdamW(groups, lr=base_lr)


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
        encoded = {k: v.to(device) for k, v in encoded.items()}
        if device == "cuda" and amp_dtype is not None:
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
        if args.amp_dtype == "fp16":
            amp_dtype = torch.float16
        elif args.amp_dtype == "bf16":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = None
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"VRAM {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB | "
            f"AMP: {args.amp_dtype}"
        )
    else:
        batch_size = args.batch_size or 8
        amp_dtype = None
        torch.set_num_threads(min(12, os.cpu_count() or 1))
        print("CPU fallback")

    effective_batch = batch_size * args.grad_accum
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
            dtype=torch.float32,   # force fp32 params so GradScaler works correctly
        )
        model.to(device)
        model.train()

        optimizer = build_optimizer_with_llrd(model, args.lr, args.weight_decay, args.llrd)
        steps_per_epoch = int(np.ceil(len(train_idx) / batch_size))
        total_steps = (steps_per_epoch // args.grad_accum) * args.epochs
        warmup_steps = int(args.warmup_ratio * total_steps)

        if args.schedule == "cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )

        # GradScaler for fp16 — essential for DeBERTa-v3 (disentangled attention overflows without it)
        use_scaler = (device == "cuda" and amp_dtype == torch.float16)
        scaler = torch.amp.GradScaler("cuda") if use_scaler else None

        fold_texts = [train_texts[i] for i in train_idx]
        fold_labels = y[train_idx]

        best_fold_f1 = -1.0
        best_oof_proba = None
        best_test_proba = None

        with tempfile.TemporaryDirectory() as tmpdir:
            best_ckpt = Path(tmpdir) / "best.pt"

            for epoch in range(args.epochs):
                model.train()
                order = np.random.permutation(len(fold_texts))
                running = 0.0
                epoch_started = time.time()
                optimizer.zero_grad(set_to_none=True)
                update_step = 0
                for step, start in enumerate(range(0, len(order), batch_size)):
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
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    if device == "cuda" and amp_dtype is not None:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            loss = loss_fn(model(**encoded).logits, batch_labels)
                    else:
                        loss = loss_fn(model(**encoded).logits, batch_labels)

                    loss = loss / args.grad_accum
                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    running += float(loss.item()) * args.grad_accum

                    if (step + 1) % args.grad_accum == 0 or (start + batch_size) >= len(order):
                        if use_scaler:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], 1.0
                        )
                        if use_scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        update_step += 1

                valid_texts = [train_texts[i] for i in valid_idx]
                valid_proba = predict_proba(
                    model, tokenizer, valid_texts, device, args.maxlen, args.eval_batch_size, amp_dtype
                )
                epoch_f1 = f1_score(y[valid_idx], valid_proba.argmax(axis=1), average="macro")

                print(
                    f"fold {fold}/{args.n_folds} epoch {epoch + 1}/{args.epochs} "
                    f"loss={running / steps_per_epoch:.4f} "
                    f"val_f1={epoch_f1:.6f} "
                    f"time={time.time() - epoch_started:.0f}s"
                    + (" *** best ***" if epoch_f1 > best_fold_f1 else "")
                )

                if epoch_f1 > best_fold_f1:
                    best_fold_f1 = epoch_f1
                    best_oof_proba = valid_proba.copy()
                    torch.save(model.state_dict(), best_ckpt)

            # reload best checkpoint for test predictions
            model.load_state_dict(torch.load(best_ckpt, map_location=device))
            best_test_proba = predict_proba(
                model, tokenizer, test_texts, device, args.maxlen, args.eval_batch_size, amp_dtype
            )

        oof[valid_idx] = best_oof_proba
        fold_scores.append(best_fold_f1)
        test_prob_sum += best_test_proba
        print(f"fold {fold} best val macro-F1={best_fold_f1:.6f}")

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
        "per_fold_f1": [float(s) for s in fold_scores],
        "elapsed_min": float((time.time() - started) / 60),
        "test_dist": np.bincount(test_pred, minlength=3).astype(int).tolist(),
        "recipe": {
            "epochs": args.epochs,
            "maxlen": args.maxlen,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": batch_size,
            "effective_batch_size": effective_batch,
            "grad_accum": args.grad_accum,
            "label_smoothing": args.label_smoothing,
            "warmup_ratio": args.warmup_ratio,
            "llrd": args.llrd,
            "schedule": args.schedule,
            "amp_dtype": args.amp_dtype,
            "eval_batch_size": args.eval_batch_size,
            "n_folds": args.n_folds,
            "class_weighted_loss": True,
            "best_checkpoint_per_fold": True,
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
