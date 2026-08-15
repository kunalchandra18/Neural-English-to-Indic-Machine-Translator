import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from .decode import batch_greedy_decode
from .vocab import EOS_IDX, PAD_IDX, SOS_IDX


def corpus_bleu_on(model, dataset, tgt_vocab, seq_length, device,
                   num_sentences=100, batch_size=64):
    """Greedy-decode a slice of `dataset` and score it against the references."""
    model.eval()
    n = min(num_sentences, len(dataset))
    references, candidates = [], []

    for start in range(0, n, batch_size):
        src, tgt = dataset[start:start + batch_size]
        preds = batch_greedy_decode(model, src, seq_length, device).cpu().numpy()
        for ref_ids, pred_ids in zip(tgt.numpy(), preds):
            references.append([tgt_vocab.decode(ref_ids, strip=(PAD_IDX, SOS_IDX, EOS_IDX))])
            candidates.append(tgt_vocab.decode(pred_ids, strip=(PAD_IDX, EOS_IDX)))

    if not candidates:
        return 0.0
    return corpus_bleu(references, candidates, smoothing_function=SmoothingFunction().method1)
