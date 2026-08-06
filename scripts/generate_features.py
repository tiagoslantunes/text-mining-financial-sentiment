"""Generate all feature arrays and save to data/processed/.

Run from project root:
    python scripts/generate_features.py

What this script generates:
    data/processed/X_w2v_sg_train.npy    Word2Vec Skip-gram (200d)
    data/processed/X_w2v_cbow_train.npy  Word2Vec CBOW (200d)
    data/processed/X_glove_train.npy     GloVe-Twitter (100d)
    data/processed/X_fin_train.npy       Financial hand-crafted (14d)
    data/processed/X_finbert_train.npy   FinBERT CLS (768d)   [~10 min CPU]
    data/processed/X_sbert_train.npy     SBERT all-mpnet (768d) [~5 min CPU]
    data/processed/X_roberta_train.npy   Twitter-RoBERTa (768d) [~10 min CPU]

"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Run from project root 
if Path(os.getcwd()).name == "scripts":
    os.chdir("..")
sys.path.insert(0, ".")

from src.features import (
    build_financial_features,
    get_bert_embeddings,
    get_sbert_embeddings,
    load_glove_twitter,
    texts_to_w2v_matrix,
    train_word2vec,
)

PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

train = pd.read_csv("data/raw/train.csv")
texts = train["text"].astype(str).tolist()
print(f"Corpus: {len(texts)} tweets\n")


def save(name: str, arr: np.ndarray) -> None:
    path = PROC / f"{name}.npy"
    np.save(path, arr.astype(np.float32))
    print(f"  saved {path}  shape={arr.shape}  dtype=float32")


# Word2Vec Skip-gram and CBOW 
print("=" * 55)
print("A. Word2Vec (Skip-gram + CBOW)")
print("=" * 55)

t0 = time.time()
print("  Training Skip-gram (sg=1)...")
w2v_sg = train_word2vec(texts, vector_size=200, window=5, sg=1, epochs=20)
X_sg = texts_to_w2v_matrix(texts, w2v_sg, dim=200)
save("X_w2v_sg_train", X_sg)

print("  Training CBOW (sg=0)...")
w2v_cbow = train_word2vec(texts, vector_size=200, window=5, sg=0, epochs=20)
X_cbow = texts_to_w2v_matrix(texts, w2v_cbow, dim=200)
save("X_w2v_cbow_train", X_cbow)
print(f"  Done in {time.time() - t0:.0f}s\n")


# GloVe-Twitter
print("=" * 55)
print("B. GloVe-Twitter-100")
print("=" * 55)

t0 = time.time()
print("  Loading GloVe-Twitter-100 (downloads ~200MB on first run)...")
glove = load_glove_twitter(dim=100)
X_glove = texts_to_w2v_matrix(texts, glove, dim=100)
save("X_glove_train", X_glove)
print(f"  Done in {time.time() - t0:.0f}s\n")


# Financial hand-crafted features
print("=" * 55)
print("C. Financial hand-crafted features (14d)")
print("=" * 55)

t0 = time.time()
X_fin = build_financial_features(train["text"])
save("X_fin_train", X_fin)
print(f"  Done in {time.time() - t0:.0f}s\n")


# Transformer CLS embeddings
print("=" * 55)
print("D. Transformer CLS embeddings (768d each)")
print("=" * 55)

encoders = [
    ("X_finbert_train",  "ProsusAI/finbert",                                False),
    ("X_sbert_train",    "all-mpnet-base-v2",                               True),
    ("X_roberta_train",  "cardiffnlp/twitter-roberta-base-sentiment-latest", False),
]

for save_name, model_id, use_sbert in encoders:
    path = PROC / f"{save_name}.npy"
    if path.exists():
        # Skip if already generated — safe to re-run the script incrementally.
        print(f"  {save_name}: already exists, skipping.")
        continue
    print(f"  Extracting {save_name} ({model_id})...")
    t0 = time.time()
    if use_sbert:
        emb = get_sbert_embeddings(texts, model_name=model_id, batch_size=32)
    else:
        emb = get_bert_embeddings(texts, model_name=model_id, batch_size=32)
    save(save_name, emb)
    print(f"  Done in {time.time() - t0:.0f}s\n")


# Summary
print("=" * 55)
print("All features generated. Files in data/processed/:")
for f in sorted(PROC.glob("*.npy")):
    arr = np.load(f)
    print(f"  {f.name:<35} shape={arr.shape}")
