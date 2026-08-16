"""English -> Bengali / Hindi neural machine translation.

`Trainer` is exposed lazily: it pulls in the evaluation stack (nltk), which
inference-only deployments never call. Importing `nmt.model` or `nmt.config`
should not require the training dependencies.
"""
from .config import Config
from .model import Seq2SeqTransformer, build_model
from .vocab import Vocab

__all__ = ["Config", "Seq2SeqTransformer", "build_model", "Trainer", "Vocab"]


def __getattr__(name):
    if name == "Trainer":
        from .trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
