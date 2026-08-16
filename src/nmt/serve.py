"""Model loading and translation shared by the demo front-ends.

Checkpoints load from `runs/` when present, otherwise from the Hub model repo, so the
same code serves a local training run and a deployment that carries no weights.
"""
import os
import pickle
from pathlib import Path

import torch

from .config import Config
from .data import encode_corpus, make_loader
from .decode import translate_loader
from .model import build_model
from .preprocessing import english
from .vocab import SOS_TOKEN

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    "Bengali": REPO_ROOT / "configs" / "bengali.yaml",
    "Hindi": REPO_ROOT / "configs" / "hindi.yaml",
}
MODEL_REPO = os.environ.get("NMT_MODEL_REPO", "kunalchandra18/cs779-nmt-en-indic")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One language resident at a time. Each model is ~270 MB of fp32 parameters once built,
# and free hosting tiers cap total memory near 1 GB, so holding both risks the process
# being killed mid-request. Switching languages costs a reload instead.
_loaded = {}


def resolve_assets(cfg, language):
    """Local files win; otherwise fetch from the Hub."""
    weights = Path(cfg.best_model_path)
    if not weights.is_absolute():
        weights = REPO_ROOT / weights
    vocab = weights.parent / f"vocab_{cfg.lang_code}.pkl"
    if weights.exists() and vocab.exists():
        return weights, vocab

    from huggingface_hub import hf_hub_download

    folder = language.lower()
    try:
        return (
            Path(hf_hub_download(MODEL_REPO, f"{folder}/best_model_{cfg.lang_code}.pth")),
            Path(hf_hub_download(MODEL_REPO, f"{folder}/vocab_{cfg.lang_code}.pkl")),
        )
    except Exception as e:
        raise FileNotFoundError(
            f"No local checkpoint for {language} at {weights}, and fetching it from "
            f"{MODEL_REPO} failed ({type(e).__name__}: {e})."
        ) from e


def load_language(language):
    """Return (model, src_vocab, tgt_vocab, cfg), evicting any other language first."""
    if language in _loaded:
        return _loaded[language]

    for other in [k for k in _loaded if k != language]:
        del _loaded[other]

    cfg = Config.load(CONFIGS[language])
    weights_path, vocab_path = resolve_assets(cfg, language)

    with open(vocab_path, "rb") as f:
        vocabs = pickle.load(f)
    src_vocab, tgt_vocab = vocabs["src"], vocabs["tgt"]

    model = build_model(cfg, len(src_vocab), len(tgt_vocab)).to(DEVICE)
    # fp16 checkpoints are upcast by load_state_dict; they decode identically to fp32.
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    _loaded[language] = (model, src_vocab, tgt_vocab, cfg)
    return _loaded[language]


def translate(text, language):
    """Translate one English sentence. Returns a message rather than raising."""
    if not text or not text.strip():
        return "Type an English sentence above to translate it."

    try:
        model, src_vocab, tgt_vocab, cfg = load_language(language)
    except FileNotFoundError as e:
        return f"⚠️ {e}"

    tokens = english.tokenize_corpus([text], n_process=1)
    if not tokens[0]:
        return "Nothing translatable in that input — try a plain English sentence."

    loader = make_loader(encode_corpus(src_vocab, tokens, cfg.seq_length), batch_size=1)
    out = translate_loader(model, loader, tgt_vocab, cfg.seq_length, DEVICE)

    # translate_loader strips PAD/EOS only, matching how the competition CSVs were scored.
    # <SOS> is a pure artifact; <UNK> stays visible as the model's honest OOV signal.
    words = [w for w in out[0].split() if w != SOS_TOKEN]
    return " ".join(words) or "(no translation produced)"
