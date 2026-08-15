from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    language: str            # "Bengali" or "Hindi"
    lang_code: str           # "bn" or "hi"

    train_pkl: str
    test_ids_pkl: str
    test_sentences_pkl: str
    output_dir: str

    seq_length: int = 55
    min_freq: int = 2
    val_split: float = 0.05
    seed: int = 42

    emb_size: int = 512
    nhead: int = 8
    hid_dim: int = 1024
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dropout: float = 0.15

    epochs: int = 15
    train_batch_size: int = 32
    inference_batch_size: int = 256
    label_smoothing: float = 0.1
    weight_decay: float = 0.01
    warmup_steps: int = 4000
    max_grad_norm: float = 1.0
    bleu_sentences: int = 40
    compile_model: bool = True

    @classmethod
    def load(cls, path):
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    @property
    def checkpoint_path(self):
        return Path(self.output_dir) / f"checkpoint_{self.lang_code}.pth"

    @property
    def best_model_path(self):
        return Path(self.output_dir) / f"best_model_{self.lang_code}.pth"
