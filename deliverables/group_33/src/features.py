"""Feature engineering: BoW, TF-IDF, Word2Vec, GloVe, Transformer encoders.

"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import gensim.downloader as api
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

sia = SentimentIntensityAnalyzer()


# ── BoW / TF-IDF ─────────────────────────────────────────────────────────────

def build_bow(train_texts, test_texts=None):
    """Binary Bag-of-Words (CountVectorizer, binary=True) — Lab 1 approach."""
    bow = CountVectorizer(binary=True, max_features=20000, min_df=2)
    X_train = bow.fit_transform(train_texts)
    X_test = bow.transform(test_texts) if test_texts is not None else None
    return bow, X_train, X_test


def build_tfidf_1g(train_texts, test_texts=None):
    """TF-IDF unigrams — Lab Extra canonical approach."""
    vec = TfidfVectorizer(ngram_range=(1, 1), max_features=20000,
                          sublinear_tf=True, min_df=2, max_df=0.8)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts) if test_texts is not None else None
    return vec, X_train, X_test


def build_tfidf_2g(train_texts, test_texts=None):
    """TF-IDF unigrams + bigrams — Lab Extra bonus challenge approach."""
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000,
                          sublinear_tf=True, min_df=2, max_df=0.8)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts) if test_texts is not None else None
    return vec, X_train, X_test


def build_tfidf_char(train_texts, test_texts=None):
    """TF-IDF character n-grams — robust to Twitter spelling noise."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          max_features=30000, sublinear_tf=True)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts) if test_texts is not None else None
    return vec, X_train, X_test


# ── Word2Vec / GloVe ─────────────────────────────────────────────────────────

def train_word2vec(sentences, vector_size=200, window=5, sg=1,
                   min_count=2, workers=4, seed=42, epochs=20):
    """Train Word2Vec on the corpus (Skip-gram or CBOW) — Lab 2 approach."""
    tokenized = [text.split() for text in sentences]
    model = Word2Vec(tokenized, vector_size=vector_size, window=window,
                     sg=sg, min_count=min_count, workers=workers,
                     seed=seed, epochs=epochs)
    return model


def doc_to_vec(text: str, model, dim: int) -> np.ndarray:
    """Mean pooling of word vectors — Mikolov et al. (2013) Lab 2 approach."""
    tokens = text.split()
    if hasattr(model, "wv"):
        vecs = [model.wv[t] for t in tokens if t in model.wv]
    else:
        vecs = [model[t] for t in tokens if t in model]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)


def texts_to_w2v_matrix(texts, model, dim: int) -> np.ndarray:
    """Convert a list of texts to a (N, dim) embedding matrix."""
    return np.vstack([doc_to_vec(t, model, dim) for t in texts])


def load_glove_twitter(dim: int = 100):
    """Load GloVe-Twitter pre-trained embeddings via gensim downloader.

    Trained on 2B tweets — ideal for financial tweet classification
    (Barbieri et al., 2020).
    """
    return api.load(f"glove-twitter-{dim}")


# ── Transformer Encoder CLS Embeddings ───────────────────────────────────────

def get_bert_embeddings(texts, model_name: str, batch_size: int = 32,
                        device: str = DEVICE) -> np.ndarray:
    """Extract [CLS] token embeddings from a transformer encoder.

    Follows Lab 4's generate_cls_embeddings() approach: extract the first
    token (index [0][0]) of the feature-extraction pipeline output.

    Args:
        texts: List of raw tweet strings.
        model_name: HuggingFace model identifier.
        batch_size: Texts per batch.
        device: 'cpu' or 'cuda'.

    Returns:
        (N, hidden_size) numpy array.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)

    return np.vstack(all_embeddings)


def get_sbert_embeddings(texts, model_name: str = "all-mpnet-base-v2",
                         batch_size: int = 32) -> np.ndarray:
    """Sentence-BERT embeddings — Reimers & Gurevych (2019)."""
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size,
                        show_progress_bar=True, normalize_embeddings=True)


# ── Financial Hand-crafted Features ─────────────────────────────────────────

def extract_financial_features(text: str) -> dict:
    """VADER sentiment + Twitter surface features + readability metrics."""
    sentiment = sia.polarity_scores(text)
    return {
        "vader_pos": sentiment["pos"],
        "vader_neg": sentiment["neg"],
        "vader_neu": sentiment["neu"],
        "vader_compound": sentiment["compound"],
        "n_cashtags": len(re.findall(r"\$[A-Z]{1,5}", text)),
        "n_hashtags": len(re.findall(r"#\w+", text)),
        "n_mentions": len(re.findall(r"@\w+", text)),
        "has_url": int(bool(re.search(r"https?://", text))),
        "word_count": len(text.split()),
        "char_count": len(text),
        "exclamation": text.count("!"),
        "question": text.count("?"),
        "all_caps_ratio": sum(1 for w in text.split() if w.isupper()) / max(len(text.split()), 1),
        "flesch_score": textstat.flesch_reading_ease(text),
    }


def build_financial_features(series: pd.Series) -> np.ndarray:
    """Apply extract_financial_features to a pandas Series, return numpy array."""
    rows = [extract_financial_features(t) for t in series]
    return pd.DataFrame(rows).values
