"""Gradio demo: translate English into Bengali or Hindi.

Run locally:  python app.py   (needs `pip install -r requirements-demo.txt`)

The Streamlit front-end in streamlit_app.py is what runs in the hosted deployment;
both share the loading and decoding logic in nmt.serve.
"""
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nmt import serve

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
    go.click(serve.translate, inputs=[src, language], outputs=out)
    src.submit(serve.translate, inputs=[src, language], outputs=out)

if __name__ == "__main__":
    demo.launch()
