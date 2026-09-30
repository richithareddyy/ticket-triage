"""NLTK text processing: normalization, stopword removal, lemmatization, sentiment."""
import os
import re
from functools import lru_cache

import nltk

NLTK_DATA = os.environ.get("NLTK_DATA", os.path.join(os.path.dirname(__file__), "..", "nltk_data"))
NLTK_PACKAGES = {
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
    "sentiment/vader_lexicon.zip": "vader_lexicon",
}

# Negations and urgency words carry signal for triage, so keep them out of the stopword list.
KEEP = {"not", "no", "nor", "down", "all", "now", "again", "very", "only"}
TOKEN_RE = re.compile(r"[a-z][a-z0-9']+")
ERROR_CODE_RE = r"(?:error\s*(?:code\s*)?[a-z]?\d{3,5}|http\s*\d{3}|\b[45]\d{2}\b|err_[a-z_]+)"
URGENCY_RE = r"\b(?:urgent|asap|immediately|critical|blocking|production|outage|down|emergency)\b"


def ensure_nltk():
    if os.path.abspath(NLTK_DATA) not in [os.path.abspath(p) for p in nltk.data.path]:
        nltk.data.path.insert(0, NLTK_DATA)
    for resource, package in NLTK_PACKAGES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, download_dir=NLTK_DATA, quiet=True)


@lru_cache(maxsize=1)
def _tools():
    ensure_nltk()
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.stem import WordNetLemmatizer

    stop = set(stopwords.words("english")) - KEEP
    return stop, WordNetLemmatizer(), SentimentIntensityAnalyzer()


def clean_text(text):
    """Lowercase, tokenize, drop stopwords, lemmatize. Error codes are collapsed to a token."""
    stop, lemmatizer, _ = _tools()
    text = re.sub(ERROR_CODE_RE, " errcode ", (text or "").lower())
    return " ".join(lemmatizer.lemmatize(t) for t in TOKEN_RE.findall(text) if t not in stop)


def sentiment(text):
    """VADER compound score in [-1, 1]; negative means an angry/frustrated customer."""
    return _tools()[2].polarity_scores(text or "")["compound"]
