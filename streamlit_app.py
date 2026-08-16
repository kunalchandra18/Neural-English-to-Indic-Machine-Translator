"""Streamlit demo: translate English into Bengali or Hindi.

Run locally:  streamlit run streamlit_app.py
Deployed at:  https://share.streamlit.io (entry point: streamlit_app.py)

Weights come from the Hugging Face model repo when `runs/` is absent, so this file
plus `src/` and `configs/` is everything the deployment needs.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nmt import serve

st.set_page_config(page_title="English → Bengali / Hindi", page_icon="🪷")

EXAMPLES = [
    "The weather is very pleasant today.",
    "The train arrives at the station in ten minutes.",
    "She is reading a book in the library.",
    "I want to learn a new language this year.",
]


@st.cache_resource(show_spinner="Loading model…")
def warm(language):
    """Cache the loaded model across reruns; Streamlit reruns the script per interaction."""
    serve.load_language(language)
    return language


st.title("English → Bengali / Hindi")
st.markdown(
    "A Pre-LN Transformer trained from scratch for the **CS779** machine translation "
    "competition at IIT Kanpur. "
    "[Code](https://github.com/kunalchandra18/Neural-English-to-Indic-Machine-Translator) · "
    "[Weights](https://huggingface.co/kunalchandra18/cs779-nmt-en-indic)"
)

language = st.radio("Translate into", ["Hindi", "Bengali"], horizontal=True)

if "text" not in st.session_state:
    st.session_state.text = EXAMPLES[0]

st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex.split()[1].title(), help=ex, use_container_width=True):
        st.session_state.text = ex

text = st.text_area("English", key="text", height=110)

if st.button("Translate", type="primary"):
    warm(language)
    with st.spinner("Translating…"):
        st.text_area("Translation", serve.translate(text, language), height=110)

st.divider()
st.caption(
    "Trained on a small corpus with greedy decoding. Everyday sentences translate well; "
    "proper nouns often do not, since a frequency-2 vocabulary cutoff leaves them out of "
    "vocabulary — the main error source identified in the report."
)
