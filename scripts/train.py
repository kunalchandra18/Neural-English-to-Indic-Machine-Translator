"""Train the Pre-LN Transformer for one language pair.

    python scripts/train.py --config configs/bengali.yaml
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from sklearn.model_selection import train_test_split

from nmt.config import Config
from nmt.data import encode_corpus, load_pickle, make_dataset, make_loader
from nmt.model import build_model
from nmt.trainer import Trainer
from nmt.vocab import Vocab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    cfg = Config.load(args.config)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data = load_pickle(cfg.train_pkl)
    src_all = data[f"source_{cfg.language}"]
    tgt_all = data[f"target_{cfg.language}"]
    test_src = load_pickle(cfg.test_sentences_pkl)

    src_train, src_val, tgt_train, tgt_val = train_test_split(
        src_all, tgt_all, test_size=cfg.val_split, random_state=cfg.seed)

    src_vocab = Vocab.build(src_train + src_val + test_src, cfg.min_freq)
    tgt_vocab = Vocab.build(tgt_train + tgt_val, cfg.min_freq)
    print(f"vocab: src {len(src_vocab)}, tgt {len(tgt_vocab)}")

    with open(Path(cfg.output_dir) / f"vocab_{cfg.lang_code}.pkl", "wb") as f:
        pickle.dump({"src": src_vocab, "tgt": tgt_vocab}, f)

    train_loader = make_loader(
        encode_corpus(src_vocab, src_train, cfg.seq_length),
        encode_corpus(tgt_vocab, tgt_train, cfg.seq_length),
        batch_size=cfg.train_batch_size, shuffle=True, drop_last=True)

    val_dataset = make_dataset(
        encode_corpus(src_vocab, src_val, cfg.seq_length),
        encode_corpus(tgt_vocab, tgt_val, cfg.seq_length))

    model = build_model(cfg, len(src_vocab), len(tgt_vocab)).to(device)
    trainer = Trainer(model, cfg, device)

    if args.resume and trainer.resume():
        print(f"resumed at epoch {trainer.start_epoch}, best bleu {trainer.best_bleu:.4f}")

    best = trainer.fit(train_loader, val_dataset, tgt_vocab)
    print(f"done. best bleu {best:.4f} -> {cfg.best_model_path}")


if __name__ == "__main__":
    main()
