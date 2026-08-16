import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .evaluate import corpus_bleu_on
from .model import generate_square_subsequent_mask
from .vocab import PAD_IDX


def noam_schedule(d_model, warmup_steps):
    """Inverse-sqrt decay with linear warmup, as in Vaswani et al. (2017)."""
    def lr_lambda(step):
        step = step + 1
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    return lr_lambda


def _strip_compile_prefix(state_dict):
    return OrderedDict(
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k, v)
        for k, v in state_dict.items()
    )


def _unwrap(model):
    return getattr(model, "_orig_mod", model)


def _should_compile(cfg, device):
    """torch.compile lowers through Triton, which needs CUDA capability 7.0+.

    Kaggle hands out Tesla P100s (6.0), where compiling raises on the first batch.
    Training there is perfectly fine uncompiled, so warn and carry on rather than die.
    """
    if not cfg.compile_model or device.type != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability(device)
    if major < 7:
        print(f"torch.compile disabled: {torch.cuda.get_device_name(device)} is compute "
              f"capability {major}.{minor}, Triton requires 7.0+. Training uncompiled.")
        return False
    return True


class Trainer:
    def __init__(self, model, cfg, device):
        self.cfg = cfg
        self.device = device
        self.model = torch.compile(model) if _should_compile(cfg, device) else model
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=cfg.label_smoothing)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1.0, betas=(0.9, 0.98),
                                     eps=1e-9, weight_decay=cfg.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, noam_schedule(cfg.emb_size, cfg.warmup_steps))
        self.scaler = torch.amp.GradScaler(device.type)
        self.start_epoch = 1
        self.best_bleu = 0.0
        self.train_losses = []
        self.global_step = 0

    def resume(self):
        path = self.cfg.checkpoint_path
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=self.device)
        _unwrap(self.model).load_state_dict(_strip_compile_prefix(ckpt["model_state_dict"]))
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_bleu = ckpt["best_bleu"]
        self.train_losses = ckpt["train_losses"]
        self.global_step = ckpt["global_step"]
        return True

    def _run_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0
        bar = tqdm(loader, desc=f"Epoch {epoch}/{self.cfg.epochs}")

        for src, tgt in bar:
            src, tgt = src.to(self.device), tgt.to(self.device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            tgt_mask = generate_square_subsequent_mask(tgt_input.shape[1], self.device)
            src_padding_mask = src == PAD_IDX

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
                logits = self.model(src, tgt_input, tgt_mask, src_padding_mask,
                                    tgt_input == PAD_IDX, src_padding_mask)
                loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1

            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.3e}")

        return total_loss / len(loader)

    def fit(self, train_loader, val_dataset, tgt_vocab, on_epoch_end=None):
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            started = time.time()
            avg_loss = self._run_epoch(train_loader, epoch)
            self.train_losses.append(avg_loss)

            bleu = corpus_bleu_on(self.model, val_dataset, tgt_vocab, self.cfg.seq_length,
                                  self.device, num_sentences=self.cfg.bleu_sentences)
            print(f"epoch {epoch} | loss {avg_loss:.4f} | bleu {bleu:.4f} "
                  f"| {time.time() - started:.0f}s")

            if bleu > self.best_bleu:
                self.best_bleu = bleu
                torch.save(_unwrap(self.model).state_dict(), self.cfg.best_model_path)
                print(f"  new best -> {self.cfg.best_model_path}")

            torch.save({
                "epoch": epoch,
                "model_state_dict": _unwrap(self.model).state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_bleu": self.best_bleu,
                "train_losses": self.train_losses,
                "global_step": self.global_step,
            }, self.cfg.checkpoint_path)

            if on_epoch_end:
                on_epoch_end(epoch)

        return self.best_bleu
