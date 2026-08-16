"""Preprocessing pipelines, imported lazily.

`english` needs spaCy and `indic` needs indic-nlp-library. Importing both eagerly
would force inference-only deployments (which tokenize English input and detokenize
target text from the vocabulary) to install a dependency they never call.
"""
from .corpus import LANG_CODES, read_split

__all__ = ["english", "indic", "read_split", "LANG_CODES"]


def __getattr__(name):
    if name in ("english", "indic"):
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
