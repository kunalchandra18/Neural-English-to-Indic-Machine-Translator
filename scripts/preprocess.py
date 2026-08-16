"""Tokenize the raw competition JSON into pickled token lists.

    python scripts/preprocess.py --train data/train.json --val data/val.json \
        --test data/test.json --out data/processed
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nmt.preprocessing import LANG_CODES, english, indic, read_split


def dump(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = read_split(args.train)
    val = read_split(args.val)
    test = read_split(args.test)

    train_out, val_out = {}, {}
    for language, code in LANG_CODES.items():
        pair = f"English-{language}"
        print(f"\n== {pair} ==")
        train_out[f"source_{language}"] = english.tokenize_corpus(train[pair]["source"])
        train_out[f"target_{language}"] = indic.tokenize_corpus(train[pair]["target"], lang=code)
        train_out[f"ids_{language}"] = train[pair]["ids"]

        val_out[f"source_{language}"] = english.tokenize_corpus(val[pair]["source"])
        val_out[f"ids_{language}"] = val[pair]["ids"]

        dump(english.tokenize_corpus(test[pair]["source"]), out / f"test_{code}_sentences.pkl")
        dump(test[pair]["ids"], out / f"test_{code}_ids.pkl")

    dump(train_out, out / "train.pkl")
    dump(val_out, out / "val.pkl")
    print(f"\nwrote processed corpora to {out}")


if __name__ == "__main__":
    main()
