import torch

from .model import generate_square_subsequent_mask
from .vocab import EOS_IDX, PAD_IDX, SOS_IDX


@torch.no_grad()
def batch_greedy_decode(model, src, max_len, device, start_symbol=SOS_IDX):
    model.eval()
    src = src.to(device)
    src_padding_mask = src == PAD_IDX
    memory = model.encode(src, src_padding_mask)

    ys = torch.full((src.shape[0], 1), start_symbol, dtype=torch.long, device=device)
    ended = torch.zeros(src.shape[0], dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.shape[1], device)
        out = model.decode(ys, memory, tgt_mask, ys == PAD_IDX, src_padding_mask)
        next_word = model.generator(out[:, -1]).argmax(dim=1)
        next_word = next_word.masked_fill(ended, PAD_IDX)
        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
        ended |= next_word == EOS_IDX
        if ended.all():
            break

    return ys[:, 1:]


def translate_loader(model, loader, tgt_vocab, max_len, device):
    """Greedy-decode every batch in `loader` into detokenized target strings."""
    outputs = []
    for batch in loader:
        preds = batch_greedy_decode(model, batch[0], max_len, device)
        for row in preds.cpu().numpy():
            outputs.append(" ".join(tgt_vocab.decode(row, strip=(PAD_IDX, EOS_IDX))))
    return outputs
