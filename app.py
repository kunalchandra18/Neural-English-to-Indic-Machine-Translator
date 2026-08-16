"""Gradio demo: translate English into Bengali or Hindi.

Run locally:      python app.py
Deploy:           push this file + requirements.txt to a Hugging Face Space (SDK: gradio)

Expects, per language, a checkpoint and the vocabulary it was trained with:

    runs/bengali/best_model_bn.pth   runs/bengali/vocab_bn.pkl
    runs/hindi/best_model_hi.pth     runs/hindi/vocab_hi.pkl

Both files are written by scripts/train.py. The vocabulary is not optional --
a checkpoint without its exact word->index map decodes to noise.
"""
import pickle
import sys
from pathlib import Path

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nmt.config import Config
from nmt.data import encode_corpus, make_loader
from nmt.decode import translate_loader
from nmt.model import build_model
from nmt.preprocessing import english
from nmt.vocab import SOS_TOKEN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIGS = {"Bengali": "configs/bengali.yaml", "Hindi": "configs/hindi.yaml"}

_loaded = {}


def load_language(language):
    """Load and cache (model, tgt_vocab, src_vocab, cfg). Raises if assets are missing."""
    if language in _loaded:
        return _loaded[language]

    cfg = Config.load(CONFIGS[language])
    vocab_path = Path(cfg.output_dir) / f"vocab_{cfg.lang_code}.pkl"
    weights_path = cfg.best_model_path

    for path, what in ((weights_path, "checkpoint"), (vocab_path, "vocabulary")):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {what} for {language}: {path}\n"
                f"Train it first:  python scripts/train.py --config {CONFIGS[language]}"
            )

    with open(vocab_path, "rb") as f:
        vocabs = pickle.load(f)
    src_vocab, tgt_vocab = vocabs["src"], vocabs["tgt"]

    model = build_model(cfg, len(src_vocab), len(tgt_vocab)).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    _loaded[language] = (model, src_vocab, tgt_vocab, cfg)
    return _loaded[language]


def translate(text, language):
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

    # translate_loader strips PAD/EOS only, matching how the competition CSVs were
    # scored. <SOS> is a pure artifact, so drop it here rather than show it to a reader.
    # <UNK> is left visible: it is the model's honest signal for an out-of-vocabulary word.
    words = [w for w in out[0].split() if w != SOS_TOKEN]
    return " ".join(words) or "(no translation produced)"


EXAMPLES = [
    ["The weather is very pleasant today.", "Hindi"],
    ["She is reading a book in the library.", "Bengali"],
    ["I want to learn a new language this year.", "Hindi"],
    ["The train arrives at the station in ten minutes.", "Bengali"],
]

with gr.Blocks(title="English → Bengali / Hindi Translator") as demo:
    gr.Markdown(
        "# English → Bengali / Hindi\n"
        "A Pre-LN Transformer trained from scratch for the CS779 machine translation "
        "competition at IIT Kanpur. Pick a target language and enter an English sentence.\n\n"
        "*Trained on a small corpus with greedy decoding — expect rough output on long or "
        "unusual sentences.*"
    )
    with gr.Row():
        with gr.Column():
            src = gr.Textbox(label="English", lines=4, placeholder="Enter an English sentence...")
            language = gr.Radio(["Hindi", "Bengali"], value="Hindi", label="Translate into")
            go = gr.Button("Translate", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="Translation", lines=4, show_copy_button=True)

    gr.Examples(examples=EXAMPLES, inputs=[src, language])
    go.click(translate, inputs=[src, language], outputs=out)
    src.submit(translate, inputs=[src, language], outputs=out)

if __name__ == "__main__":
    demo.launch()
