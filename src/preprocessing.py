"""Text preprocessing pipeline for financial tweet classification.

All techniques follow the professor's canonical implementation from Lab 1
(SnowballStemmer, WordNetLemmatizer) and Lab Extra (TweetTokenizer for tweets).
"""

import re
import unicodedata
import os
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer


def _nltk_download_dir():
    data_dirs = os.environ.get("NLTK_DATA")
    if not data_dirs:
        return None
    return data_dirs.split(os.pathsep)[0]


def _resource_exists(resource_paths: tuple[str, ...]) -> bool:
    for resource_path in resource_paths:
        try:
            nltk.data.find(resource_path)
            return True
        except LookupError:
            continue
    return False


def _ensure_nltk_resource(resource_paths: tuple[str, ...], package_name: str) -> None:
    if _resource_exists(resource_paths):
        return
    nltk.download(package_name, quiet=True, download_dir=_nltk_download_dir())
    if not _resource_exists(resource_paths):
        paths = ", ".join(resource_paths)
        raise LookupError(f"NLTK package '{package_name}' was downloaded but not found at {paths}")


_ensure_nltk_resource(("corpora/stopwords", "corpora/stopwords.zip"), "stopwords")
_ensure_nltk_resource(("corpora/wordnet", "corpora/wordnet.zip"), "wordnet")

# Pre-instantiate (expensive objects, reuse across calls)
STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer_snow = SnowballStemmer("english")
tweet_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

# Negations preserved following Manning et al. (2008) sentiment analysis guidance
KEEP_NEGATIONS = {"not", "no", "never", "neither", "nor", "none"}


def clean_twitter_noise(text: str) -> str:
    """Remove Twitter-specific noise (URLs, mentions, cashtags, hashtags, HTML entities).

    Go et al. (2009) show these elements add noise without predictive value.
    Cashtags are mapped to a TICKER_ token to preserve financial signal.
    """
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"\$([A-Za-z]{1,5})\b", lambda m: f" TICKER_{m.group(1).upper()} ",text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = re.sub(r"&amp;|&lt;|&gt;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    """NFKD Unicode normalization + ASCII conversion + lowercase.
    Contractions are expanded before stripping apostrophes to preserve negations.
    """
    text = text.replace("’", "'").replace("‘", "'")

    text = re.sub(r"won't", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"can't", "can not", text, flags=re.IGNORECASE)
    text = re.sub(r"n't", " not", text, flags=re.IGNORECASE)

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.lower()


def remove_stopwords(tokens: list) -> list:
    """Remove stopwords, preserving negations (Manning et al., 2008)."""
    return [t for t in tokens if t not in STOP_WORDS or t in KEEP_NEGATIONS]


def lemmatize_tokens(tokens: list) -> list:
    """WordNetLemmatizer — same as professor's Lab 1 implementation."""
    return [lemmatizer.lemmatize(t) for t in tokens]


_STRUCTURED_PREFIXES = ("ticker_", "url", "user")

def stem_tokens(tokens: list) -> list:
    """SnowballStemmer — same as professor's Lab 1 implementation.

    Structured tokens introduced by clean_twitter_noise (ticker_*, url, user)
    are passed through unchanged so the stemmer cannot corrupt them.
    """
    return [
        t if t.startswith(_STRUCTURED_PREFIXES) else stemmer_snow.stem(t)
        for t in tokens
    ]


def tokenize_tweet(text: str) -> list:
    """TweetTokenizer: handles emoticons, reduces character repetitions."""
    return tweet_tokenizer.tokenize(text)


def full_pipeline(
    text: str,
    use_lemmatize: bool = True,
    use_stem: bool = False,
) -> str:
    """Full preprocessing pipeline: clean → normalize → tokenize → filter → reduce.

    Args:
        text: Raw tweet string.
        use_lemmatize: Apply WordNet lemmatization.
        use_stem: Apply Snowball stemming (overrides lemmatize).

    Returns:
        Space-joined token string ready for vectorization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = clean_twitter_noise(text)
    text = normalize_text(text)
    tokens = tokenize_tweet(text)
    tokens = [t for t in tokens if len(t) > 1 and re.match(r"[a-z_]", t)]
    tokens = remove_stopwords(tokens)
    if use_stem:
        tokens = stem_tokens(tokens)
    elif use_lemmatize:
        tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


def apply_pipeline(series, use_lemmatize: bool = True, use_stem: bool = False):
    """Vectorised application of full_pipeline to a pandas Series."""
    return series.apply(lambda x: full_pipeline(x, use_lemmatize=use_lemmatize, use_stem=use_stem))
