# English → Bengali / Hindi Neural Machine Translation

A from-scratch NMT system built for the **CS779 (Statistical Natural Language Processing) course competition** at IIT Kanpur. Four weeks of architecture search — GRU → Bi-LSTM with attention → vanilla Transformer → stacked Bi-GRU — converged on a **Pre-LN Transformer with tied embeddings**, which is what this repository implements.

**Final leaderboard: rank 35** · BLEU **0.162** · chrF **0.417** · ROUGE-L **0.445**

---

## Results

### Test set (final phase)

| Model | BLEU | chrF | ROUGE-L |
|---|---|---|---|
| Pre-LN Transformer + BPE | 0.059 | 0.192 | 0.218 |
| **Pre-LN Transformer, word-level (submitted)** | **0.162** | **0.417** | **0.445** |

The BPE variant scored *lower* despite a lower training loss — the tokenizer produced poor segment boundaries on Bengali, so the loss curve was measuring the wrong thing. It was dropped for the final submission.

### Validation set (development phase)

| Week | Architecture | BLEU | chrF | ROUGE-L |
|---|---|---|---|---|
| 1 | Vanilla GRU (baseline) | 0.058 | 0.233 | 0.319 |
| 2 | Bi-LSTM + Bahdanau attention | 0.078 | 0.264 | 0.277 |
| 3 | Vanilla Transformer (Post-LN) | 0.073 | 0.280 | 0.340 |
| 4 | Stacked Bi-GRU + global attention | 0.102 | 0.326 | 0.362 |

The full write-up, including error analysis, is in [`report/NMT_Report.pdf`](report/NMT_Report.pdf).

---

## Architecture

The final model is a 6-layer encoder / 6-layer decoder Transformer with three deliberate departures from the original 2017 design:

- **Pre-Layer Normalization** — LayerNorm is applied *before* each attention and feed-forward block. Post-LN training was volatile and repeatedly diverged at high learning rates; Pre-LN made the warmup schedule usable ([Wang et al., 2019](https://arxiv.org/abs/1906.01787)).
- **GELU activations** instead of ReLU, for smoother optimization through a deep stack ([Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415)).
- **Weight tying** between the target embedding matrix and the output projection, which cut the parameter count and acted as the main defense against overfitting on a small corpus ([Press & Wolf, 2017](https://arxiv.org/abs/1608.05859)).

| Hyperparameter | Value |
|---|---|
| Embedding / model dim | 512 |
| Attention heads | 8 |
| Feed-forward dim | 1024 |
| Encoder / decoder layers | 6 / 6 |
| Dropout | 0.15 |
| Max sequence length | 55 |
| Optimizer | AdamW (β = 0.9/0.98, wd = 0.01) |
| LR schedule | Noam inverse-sqrt, 4000 warmup steps |
| Loss | Cross-entropy, label smoothing 0.1 |
| Precision | Mixed (fp16 autocast + grad scaler) |
| Decoding | Greedy |

---

## Data

| Pair | Train | Val | Test | Avg. length | Src vocab | Tgt vocab |
|---|---|---|---|---|---|---|
| English–Bengali | 68,849 | 9,836 | 19,672 | 16.8 | 31,920 | 37,921 |
| English–Hindi | 149,646 | 21,379 | 42,757 | 31.4 | 33,366 | 31,680 |

Preprocessing is asymmetric by design, because the two sides fail in different ways:

**English** — URL/HTML/email stripping, contraction expansion (`n't` → ` not`), spaCy tokenization, then aggressive junk filtering. Tracking codes and long digit runs are discarded; they inflate the vocabulary and carry no translatable meaning.

**Bengali / Hindi** — noise removal, then **Unicode normalization via `IndicNormalizerFactory`** before tokenizing with `indic_tokenize.trivial_tokenize`. This step matters more than it looks: Indic scripts admit several encodings of the same vowel or conjunct, and without normalization the same word appears as multiple distinct vocabulary entries.

Both sides use a frequency-2 vocabulary cutoff with `<PAD> <SOS> <EOS> <UNK>` reserved at indices 0–3.

---

## Repository layout

```
src/nmt/
├── config.py              # dataclass-backed YAML config
├── vocab.py               # Vocab, special tokens, encode/decode
├── model.py               # PositionalEncoding, Seq2SeqTransformer
├── data.py                # encoding + DataLoader construction
├── decode.py              # batched greedy decoding
├── evaluate.py            # corpus BLEU
├── trainer.py             # training loop, AMP, checkpointing, resume
└── preprocessing/
    ├── corpus.py          # competition JSON reader
    ├── english.py         # spaCy pipeline + junk filtering
    └── indic.py           # Indic normalization + tokenization

app.py                     # Gradio demo (English -> Bengali / Hindi)
scripts/                   # preprocess.py, train.py, translate.py
configs/                   # bengali.yaml, hindi.yaml
notebooks/                 # original Kaggle notebooks, as submitted
report/                    # competition report (PDF)
predictions/               # best-model test predictions
```

The two original training notebooks were byte-identical apart from the language name, so the refactor collapses them into one pipeline parameterized by a config file.

---

## Usage

```bash
pip install -r requirements-train.txt
python -m spacy download en_core_web_sm
pip install -e .
```

(The root `requirements.txt` holds the demo's dependencies, because Streamlit Community
Cloud installs that file from the repository root.)

Tokenize the raw competition JSON:

```bash
python scripts/preprocess.py --train data/train.json --val data/val.json --test data/test.json --out data/processed
```

Train one language pair (`--resume` picks up from the last checkpoint):

```bash
python scripts/train.py --config configs/bengali.yaml
```

Decode the test set into a submission CSV:

```bash
python scripts/translate.py --config configs/bengali.yaml --out answers_bn.csv
```

Training was done on a single Kaggle T4. Mixed precision and `torch.compile` are on by default; set `compile_model: false` in the config if your PyTorch build doesn't support it.

---

## Retrained results

The competition checkpoints were unusable (see [Live demo](#live-demo)), so both pairs were
retrained from this code on a Kaggle P100 — 15 epochs each, 150 minutes total at 8.1
steps/sec. Vocabularies came out at 31920/37921 (Bengali) and 33366/31680 (Hindi),
matching the sizes reported for the original submission.

Hindi, final epochs — the trainer keeps the best-BLEU checkpoint, not the last one:

| Epoch | Loss | BLEU |
|-------|--------|--------|
| 11    | 3.2906 | 0.2235 |
| 12    | 3.2123 | **0.2334** |
| 13    | 3.1433 | 0.2188 |
| 14    | 3.0832 | 0.2208 |
| 15    | 3.0295 | 0.2205 |

BLEU is measured on 40 validation sentences (`bleu_sentences`), so it indicates the trend
rather than a headline score. Loss keeps falling after epoch 12 while BLEU flattens, which
is where mild overfitting starts.

Sample output:

| English | Hindi | Bengali |
|---------|-------|---------|
| The weather is very pleasant today. | आज मौसम बहुत ही सुखद है | আজকের আবহাওয়া খুব সুন্দর |
| The train arrives at the station in ten minutes. | स्टेशन में ट्रेन दस मिनट में आती है | দশ মিনিটের মধ্যে ট্রেন স্টেশনে আসে । |

Everyday sentences come out well. Proper nouns do not: "My name is Kunal" mistranslates
*Kunal*, since it falls outside a vocabulary built with `min_freq: 2` — the same
out-of-vocabulary weakness the report identifies as the main error source.

The fp16 checkpoints decode identically to fp32 on every sentence tested, so the demo
ships the half-precision copies at roughly 60% of the size.

## Live demo

Trained weights are published at
**[huggingface.co/kunalchandra18/cs779-nmt-en-indic](https://huggingface.co/kunalchandra18/cs779-nmt-en-indic)**.

Two front-ends share the loading and decoding logic in [`src/nmt/serve.py`](src/nmt/serve.py):

| File | Framework | Use |
|---|---|---|
| [`streamlit_app.py`](streamlit_app.py) | Streamlit | `streamlit run streamlit_app.py` — this is what the hosted deployment runs |
| [`app.py`](app.py) | Gradio | `python app.py`, after `pip install -r requirements-demo.txt` |

Both look for local weights first and fall back to downloading from the model repo, so a
deployment carries no checkpoints of its own. Only one language stays resident at a time —
each model is ~270 MB once built, and free hosting tiers cap memory near 1 GB.

Weights and vocabulary are loaded as a pair:

```
runs/bengali/best_model_bn.pth   runs/bengali/vocab_bn.pkl
runs/hindi/best_model_hi.pth     runs/hindi/vocab_hi.pkl
```

**The vocabulary file is not optional.** A checkpoint is meaningless without the exact word→index mapping it was trained against — the embedding rows are addressed by index, so a vocabulary that differs by even one token decodes to noise. The original competition notebooks rebuilt vocabularies in memory and never saved them, which left those checkpoints unusable; `scripts/train.py` writes `vocab_{code}.pkl` next to every checkpoint so that cannot happen again.

## What I'd do differently

- **Subword tokenization, done properly.** Bengali is morphologically rich and the word-level vocabulary leaves a long OOV tail — the single biggest source of errors in the final model. My BPE attempt failed on segmentation, not on the idea.
- **Beam search.** Everything here decodes greedily because beam search was too slow to fit the submission budget. It is close to free accuracy.
- **Separate models per language pair.** One architecture served both pairs throughout, despite Hindi having twice the data and sentences nearly twice as long.
- **Word dropout** during training, to force the model to cope with unseen tokens rather than leaning on rare-word memorization.

A late run of the same architecture at 15 epochs (rather than 8) scored materially better but landed after the deadline; those predictions are in [`predictions/`](predictions/).

---

## References

1. Vaswani et al. (2017), *Attention Is All You Need*
2. Wang et al. (2019), *On Layer Normalization in the Transformer Architecture*
3. Press & Wolf (2017), *Using the Output Embedding to Improve Language Models*
4. Loshchilov & Hutter (2017), *Decoupled Weight Decay Regularization*
5. Hendrycks & Gimpel (2016), *Gaussian Error Linear Units (GELUs)*

## Author

**Kunal Chandra** (240580) — Indian Institute of Technology Kanpur, 2025

Licensed under [MIT](LICENSE).
