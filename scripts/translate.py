"""Greedy-decode the test set with a trained checkpoint into a submission CSV.

    python scripts/translate.py --config configs/bengali.yaml --out answers_bn.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from nmt.config import Config
from nmt.data import encode_corpus, load_pickle, make_loader
from nmt.decode import translate_loader
from nmt.model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = Config.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocabs = load_pickle(Path(cfg.output_dir) / f"vocab_{cfg.lang_code}.pkl")
    src_vocab, tgt_vocab = vocabs["src"], vocabs["tgt"]

    test_src = load_pickle(cfg.test_sentences_pkl)
    test_ids = load_pickle(cfg.test_ids_pkl)
    loader = make_loader(encode_corpus(src_vocab, test_src, cfg.seq_length),
                         batch_size=cfg.inference_batch_size)

    model = build_model(cfg, len(src_vocab), len(tgt_vocab)).to(device)
    model.load_state_dict(torch.load(args.checkpoint or cfg.best_model_path, map_location=device))

    translations = translate_loader(model, loader, tgt_vocab, cfg.seq_length, device)
    pd.DataFrame({"ID": list(test_ids), "Translation": translations}).to_csv(args.out, index=False)
    print(f"wrote {len(translations)} translations to {args.out}")


if __name__ == "__main__":
    main()
