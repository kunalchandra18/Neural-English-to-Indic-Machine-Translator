"""Streamlit demo: translate English into Hindi or Bengali.

Run locally:  streamlit run streamlit_app.py

Weights come from the Hugging Face model repo when `runs/` is absent, so this file
plus `src/` and `configs/` is everything the deployment needs.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nmt import serve

st.set_page_config(page_title="English → Hindi / Bengali Translator", page_icon="🪷")

PLACEHOLDER = "Choose an example…"
EXAMPLES = [
    "The weather is very pleasant today.",
    "The train arrives at the station in ten minutes.",
    "She is reading a book in the library.",
    "I want to learn a new language this year.",
    "The children are playing in the garden.",
    "This book is very interesting.",
]

st.markdown(
    """
    <style>
      .translation-box {
        border: 1px solid rgba(128,128,128,0.35);
        border-radius: 0.5rem;
        padding: 1.1rem 1.25rem;
        min-height: 5.5rem;
        font-size: 1.45rem;
        line-height: 2.1rem;
      }
      .translation-empty { opacity: 0.45; font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def warm(language):
    """Cache the loaded model across reruns; Streamlit reruns the script per interaction."""
    serve.load_language(language)
    return language


def use_example():
    if st.session_state.example != PLACEHOLDER:
        st.session_state.text = st.session_state.example


st.title("English → Hindi / Bengali")
st.write(
    "Type an English sentence and get it translated into Hindi or Bengali. "
    "The model is a Transformer trained from scratch on parallel text — no pretrained "
    "translation system is used."
)

language = st.radio("Translate into", ["Hindi", "Bengali"], horizontal=True)

st.selectbox(
    "Examples",
    [PLACEHOLDER] + EXAMPLES,
    key="example",
    on_change=use_example,
    label_visibility="collapsed",
)

text = st.text_area(
    "English",
    key="text",
    height=120,
    placeholder="Enter an English sentence…",
)

translate_clicked = st.button("Translate", type="primary", use_container_width=True)

st.markdown("**Translation**")
if translate_clicked:
    with st.spinner(f"Loading the {language} model…" if language not in serve._loaded else "Translating…"):
        warm(language)
        result = serve.translate(text, language)
    st.markdown(f'<div class="translation-box">{result}</div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="translation-box translation-empty">The translation will appear here.</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    "Trained on a limited parallel corpus using greedy decoding. Everyday sentences "
    "translate well; rare words and proper nouns often do not, because the vocabulary "
    "keeps only words seen at least twice in training."
)
st.caption(
    "[Source code](https://github.com/kunalchandra18/Neural-English-to-Indic-Machine-Translator) · "
    "[Model weights](https://huggingface.co/kunalchandra18/cs779-nmt-en-indic)"
)
