import re

import spacy
from tqdm import tqdm

CONTRACTIONS = [
    (r"\bcan['’]t\b", "cannot"),
    (r"\bwon['’]t\b", "will not"),
    (r"n['’]t", " not"),
    (r"['’]re", " are"),
    (r"['’]s", " is"),
    (r"['’]d", " would"),
    (r"['’]ll", " will"),
    (r"['’]ve", " have"),
    (r"['’]m", " am"),
]

_LONG_NUMBER = re.compile(r"^\d{6,}$")
_ALNUM_JUNK = re.compile(r"^[\da-zA-Z]*\d+[a-zA-Z]+[\da-zA-Z]*$")
_NUMBER = re.compile(r"^[+\-]?\d+(\.\d+)?$")
_WORD = re.compile(r"^[a-zA-Z][a-zA-Z'\-.]*[a-zA-Z]$")


def expand_contractions(text):
    for pattern, repl in CONTRACTIONS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def clean(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[\r\n\t]", " ", text)
    text = expand_contractions(text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _keep(token):
    """Drop punctuation, currency and alphanumeric junk; keep plain words and numbers."""
    tok = token.text.lower().strip().strip("-")
    if not tok:
        return None
    if token.is_punct or token.is_space or token.is_quote or token.is_currency:
        return None
    if _LONG_NUMBER.match(tok) or _ALNUM_JUNK.match(tok):
        return None
    if _NUMBER.match(tok) or _WORD.match(tok):
        return tok
    return None


_nlp = None


def get_nlp():
    """Load the spaCy pipeline once; reloading per call dominates serving latency."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger", "lemmatizer"])
        _nlp.max_length = 2_000_000
    return _nlp


def tokenize_corpus(texts, batch_size=1000, n_process=4):
    nlp = get_nlp()
    texts = [clean(t) for t in texts]

    out = []
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size, n_process=n_process),
                    total=len(texts), desc="english"):
        out.append([t for t in (_keep(tok) for tok in doc) if t])
    return out
