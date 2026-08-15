import re

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
from tqdm import tqdm

DEVANAGARI = r"ऀ-ॿ"
BENGALI = r"ঀ-৿"

_NOISE = re.compile(rf"[^{DEVANAGARI}{BENGALI}0-9\s]")
_VALID_TOKEN = re.compile(rf"^[{DEVANAGARI}{BENGALI}0-9]+$")

_factory = IndicNormalizerFactory()


def clean(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = _NOISE.sub(" ", text)
    text = re.sub(r"[\r\n\t]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_corpus(texts, lang="hi"):
    """Normalize Indic orthography before tokenizing, so variant spellings collapse."""
    normalizer = _factory.get_normalizer(lang)
    out = []
    for text in tqdm(texts, desc=lang):
        tokens = indic_tokenize.trivial_tokenize(normalizer.normalize(clean(text)), lang)
        out.append([t for t in (tok.strip() for tok in tokens) if t and _VALID_TOKEN.match(t)])
    return out
