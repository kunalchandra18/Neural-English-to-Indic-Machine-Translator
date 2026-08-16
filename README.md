# English → Bengali / Hindi Neural Machine Translation

A neural machine translation system built from scratch — no pretrained translation models, no fine-tuning. Trained end to end on parallel text for two low-resource language pairs.

An architecture search across four designs (GRU → Bi-LSTM with attention → vanilla Transformer → stacked Bi-GRU) converged on a **Pre-LN Transformer with tied embeddings**, which is what this repository implements.

**[▶ Try the live demo](https://neural-english-to-indic-machine-translator.streamlit.app)** · **[Model weights](https://huggingface.co/kunalchandra18/cs779-nmt-en-indic)** · **[Technical report](report/NMT_Report.pdf)**

---

## Try it

**[neural-english-to-indic-machine-translator.streamlit.app](https://neural-english-to-indic-machine-translator.streamlit.app)** — no installation, pick Hindi or Bengali and type an English sentence.

**Locally:**

```bash
git clone https://github.com/kunalchandra18/Neural-English-to-Indic-Machine-Translator.git
cd Neural-English-to-Indic-Machine-Translator
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Weights download automatically from the [Hugging Face model repo](https://huggingface.co/kunalchandra18/cs779-nmt-en-indic) on first run (~350 MB), so there is nothing else to set up.

Sample output:

| English | Hindi | Bengali |
|---|---|---|
| The weather is very pleasant today. | आज मौसम बहुत ही सुखद है | আজকের আবহাওয়া খুব সুন্দর |
| The train arrives at the station in ten minutes. | स्टेशन में ट्रेन दस मिनट में आती है | দশ মিনিটের মধ্যে ট্রেন স্টেশনে আসে । |
| I want to learn a new language this year. | इस साल नई भाषा सीखना चाहती है | এই বছরের নতুন ভাষা জানতে চায় । |

Everyday sentences translate well. Proper nouns often do not — a `min_freq: 2` vocabulary cutoff leaves them out of vocabulary, which the report identifies as the dominant error source.

---

## Results

### Test set

| Model | BLEU | chrF | ROUGE-L |
|---|---|---|---|
| Pre-LN Transformer + BPE | 0.059 | 0.192 | 0.218 |
| **Pre-LN Transformer, word-level (submitted)** | **0.162** | **0.417** | **0.445** |

The BPE variant scored *lower* despite a lower training loss — the tokenizer produced poor segment boundaries on Bengali, so the loss was measuring the wrong thing. It was dropped for the final submission.

### Validation set, by week

| Week | Architecture | BLEU | chrF | ROUGE-L |
|---|---|---|---|---|
| 1 | Vanilla GRU (baseline) | 0.058 | 0.233 | 0.319 |
| 2 | Bi-LSTM + Bahdanau attention | 0.078 | 0.264 | 0.277 |
| 3 | Vanilla Transformer (Post-LN) | 0.073 | 0.280 | 0.340 |
| 4 | Stacked Bi-GRU + global attention | 0.102 | 0.326 | 0.362 |

Week 3 is the interesting row: the Transformer beat the Bi-LSTM on chrF and ROUGE-L while scoring *lower* on BLEU. BLEU rewards exact n-gram matches, so a model that is broadly right but phrased differently can move the three metrics in opposite directions.

The full write-up, including error analysis, is in [`report/NMT_Report.pdf`](report/NMT_Report.pdf).

---

## Architecture

A 6-layer encoder / 6-layer decoder Transformer with three deliberate departures from the original 2017 design:

- **Pre-Layer Normalization** — LayerNorm applied *before* each attention and feed-forward block. Post-LN training was volatile and repeatedly diverged at high learning rates; Pre-LN made the warmup schedule usable ([Wang et al., 2019](https://arxiv.org/abs/1906.01787)).
- **GELU activations** instead of ReLU, for smoother optimization through a deep stack ([Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415)).
- **Weight tying** between the target embedding matrix and the output projection, which cut the parameter count and was the main defense against overfitting on a small corpus ([Press & Wolf, 2017](https://arxiv.org/abs/1608.05859)).

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

**Bengali / Hindi** — noise removal, then **Unicode normalization via `IndicNormalizerFactory`** before tokenizing with `indic_tokenize.trivial_tokenize`. This matters more than it looks: Indic scripts admit several encodings of the same vowel or conjunct, and without normalization one word becomes several vocabulary entries.

Both sides use a frequency-2 cutoff with `<PAD> <SOS> <EOS> <UNK>` reserved at indices 0–3.

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
├── serve.py               # model loading + translation for the demos
└── preprocessing/
    ├── corpus.py          # competition JSON reader
    ├── english.py         # spaCy pipeline + junk filtering
    └── indic.py           # Indic normalization + tokenization

streamlit_app.py           # the demo interface
scripts/                   # preprocess.py, train.py, translate.py
configs/                   # bengali.yaml, hindi.yaml
notebooks/                 # original Kaggle notebooks + retraining notebook
report/                    # technical report (PDF)
predictions/               # best-model test predictions
```

The two original training notebooks were byte-identical apart from the language name, so the refactor collapses them into one pipeline parameterized by a config file.

---

## Training

```bash
pip install -r requirements-train.txt
python -m spacy download en_core_web_sm
pip install -e .
```

Tokenize the raw competition JSON:

```bash
python scripts/preprocess.py --train data/train.json --val data/val.json --test data/test.json --out data/processed
```

Train one language pair (`--resume` continues from the last checkpoint):

```bash
python scripts/train.py --config configs/bengali.yaml
```

Decode the test set into a submission CSV:

```bash
python scripts/translate.py --config configs/bengali.yaml --out answers_bn.csv
```

To reproduce the published weights on a free GPU, run [`notebooks/kaggle_retrain.ipynb`](notebooks/kaggle_retrain.ipynb) on Kaggle. It trains both pairs, verifies each checkpoint against its vocabulary, and packages the artifacts. The published run took 150 minutes on a P100 at 8.1 steps/sec.

### Checkpoints and vocabularies travel together

`scripts/train.py` writes `vocab_{code}.pkl` beside every checkpoint:

```
runs/bengali/best_model_bn.pth   runs/bengali/vocab_bn.pkl
runs/hindi/best_model_hi.pth     runs/hindi/vocab_hi.pkl
```

**This pairing is not optional.** Embedding rows are addressed by index, so a checkpoint loaded against a vocabulary that differs by even one token decodes to noise. The original competition notebooks rebuilt vocabularies in memory and never saved them, which left those checkpoints permanently unusable — the reason this repository ships a retraining notebook rather than the original weights.

---

## What I'd do differently

- **Subword modelling done properly.** The BPE attempt failed on segmentation quality, not on the premise. Bengali is morphologically rich and a word-level vocabulary handles it badly.
- **Beam search.** Greedy decoding was chosen purely for speed under a submission deadline.
- **Separate tuning per language pair.** One configuration served both, yet Hindi sentences average nearly twice the length of Bengali ones.
- **Word dropout**, to force the model to cope with the out-of-vocabulary words that dominate its errors.

---

## Author

**Kunal Chandra** (240580) — Indian Institute of Technology Kanpur, 2025

Licensed under [MIT](LICENSE).
