import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import PAD_IDX


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.15, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class Seq2SeqTransformer(nn.Module):
    """Pre-LN encoder-decoder Transformer with GELU and tied target embeddings."""

    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead,
                 src_vocab_size, tgt_vocab_size, dim_feedforward, dropout=0.15):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=F.gelu, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=F.gelu, batch_first=True, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size, padding_idx=PAD_IDX)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=PAD_IDX)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.emb_size = emb_size

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.generator.weight = self.tgt_tok_emb.weight

    def forward(self, src, tgt, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        memory = self.encode(src, src_padding_mask)
        output = self.decode(tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(output)

    def encode(self, src, src_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        return self.transformer_encoder(src_emb, None, src_padding_mask)

    def decode(self, tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size))
        return self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)


def generate_square_subsequent_mask(size, device):
    """Boolean causal mask (True = blocked), matching the bool padding masks."""
    return torch.triu(torch.ones((size, size), device=device, dtype=torch.bool), diagonal=1)


def build_model(cfg, src_vocab_size, tgt_vocab_size):
    return Seq2SeqTransformer(
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        emb_size=cfg.emb_size,
        nhead=cfg.nhead,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dim_feedforward=cfg.hid_dim,
        dropout=cfg.dropout,
    )
