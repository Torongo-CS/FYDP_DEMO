"""Training and evaluation building blocks shared by CL, local-only and FL.

All images live on the GPU as one uint8 tensor; batches are cut by index and
augmented on the GPU, so the 4 Kaggle CPUs are never the bottleneck.
Batch order and augmentation are drawn from a generator seeded by
(seed, epoch/round, client), so a resumed run repeats exactly the same data.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Bank:
    """Images (N,S,S,3 uint8) + metadata (N,D float) + labels, on one device."""

    def __init__(self, images: np.ndarray, meta: np.ndarray, labels: np.ndarray, device, crop: int):
        self.device = torch.device(device)
        self.x = torch.from_numpy(np.ascontiguousarray(images)).to(self.device)
        self.m = torch.from_numpy(meta.astype(np.float32)).to(self.device)
        self.y = torch.from_numpy(labels.astype(np.int64)).to(self.device)
        self.crop = crop
        self.size = images.shape[1]

    def with_meta(self, meta: np.ndarray) -> "Bank":
        b = Bank.__new__(Bank)
        b.device, b.x, b.y, b.crop, b.size = self.device, self.x, self.y, self.crop, self.size
        b.m = torch.from_numpy(meta.astype(np.float32)).to(self.device)
        return b

    def batch(self, idx: torch.Tensor, train: bool, gen: torch.Generator | None = None):
        x = self.x[idx].permute(0, 3, 1, 2).float().div_(255)  # B,3,S,S
        B, S, c = x.shape[0], self.size, self.crop
        if train:
            off = torch.randint(0, S - c + 1, (B, 2), generator=gen).tolist()
            x = torch.stack([x[i, :, o[0]:o[0] + c, o[1]:o[1] + c] for i, o in enumerate(off)])
            flip_h = (torch.rand(B, generator=gen) < 0.5).view(B, 1, 1, 1).to(x.device)
            flip_v = (torch.rand(B, generator=gen) < 0.5).view(B, 1, 1, 1).to(x.device)
            rot = torch.randint(0, 4, (B,), generator=gen).to(x.device)
            bright = 1 + (torch.rand(B, 1, 1, 1, generator=gen) - 0.5) * 0.2
            contrast = 1 + (torch.rand(B, 1, 1, 1, generator=gen) - 0.5) * 0.2
            x = torch.where(flip_h, x.flip(3), x)
            x = torch.where(flip_v, x.flip(2), x)
            rots = torch.stack([torch.rot90(x, k, (2, 3)) for k in range(4)])  # 4,B,3,c,c
            x = rots[rot, torch.arange(B, device=x.device)]
            mu = x.mean((1, 2, 3), keepdim=True)
            x = ((x - mu) * contrast.to(x.device) + mu) * bright.to(x.device)
            x = x.clamp_(0, 1)
        else:
            o = (S - c) // 2
            x = x[:, :, o:o + c, o:o + c]
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        return x, self.m[idx], self.y[idx]


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse class frequency, scaled to mean 1 over the classes present."""
    cnt = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = np.where(cnt > 0, 1.0 / np.maximum(cnt, 1), 0.0)
    present = cnt > 0
    w[present] = w[present] / w[present].mean()
    return torch.tensor(w, dtype=torch.float32)


def make_gen(*keys: int) -> torch.Generator:
    g = torch.Generator()
    seed = 0
    for k in keys:
        seed = (seed * 1_000_003 + int(k) + 7) % (2 ** 62)
    g.manual_seed(seed)
    return g


def cosine_lr(base: float, step: int, total: int, warmup: int = 0) -> float:
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def make_optimizer(model: nn.Module, lr: float, cfg) -> torch.optim.Optimizer:
    # SGD with momentum for every method (CL and FL alike): FedNova and SCAFFOLD
    # are defined for SGD steps, and one optimizer keeps the comparison fair.
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum,
                           weight_decay=cfg.weight_decay, nesterov=False)


def train_epochs(model, bank: Bank, idx: np.ndarray, epochs: int, lr: float, cfg, cw: torch.Tensor,
                 gen_keys: tuple, prox_ref: dict | None = None, mu: float = 0.0,
                 scaffold_c: dict | None = None, scaffold_ci: dict | None = None,
                 use_meta: bool = True, opt: torch.optim.Optimizer | None = None) -> int:
    """Train in place for `epochs` passes over `idx`. Returns the number of optimizer steps.

    A fresh optimizer is made unless `opt` is given (CL keeps one across epochs;
    FL starts a fresh one every round, the usual practice)."""
    dev = bank.device
    model.train()
    if opt is None:
        opt = make_optimizer(model, lr, cfg)
    for gp in opt.param_groups:
        gp["lr"] = lr
    amp = dev.type == "cuda" and cfg.amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    cw = cw.to(dev)
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    steps = 0
    params = dict(model.named_parameters())
    for ep in range(epochs):
        gen = make_gen(*gen_keys, ep)
        torch.manual_seed(int(torch.randint(0, 2 ** 62, (1,), generator=make_gen(*gen_keys, ep, 1))))
        perm = idx_t[torch.randperm(len(idx_t), generator=gen)]
        for s in range(0, len(perm), cfg.batch_size):
            b = perm[s:s + cfg.batch_size]
            if len(b) < 2:
                continue
            x, m, y = bank.batch(b.to(dev), train=True, gen=gen)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                logits = model(x, m) if use_meta else model(x)
                loss = F.cross_entropy(logits.float(), y, weight=cw)
            if prox_ref is not None and mu > 0:
                prox = sum(((params[k] - prox_ref[k]) ** 2).sum() for k in prox_ref)
                loss = loss + 0.5 * mu * prox
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if scaffold_c is not None:
                scaler.unscale_(opt)
                for k, p in params.items():
                    if p.grad is not None and k in scaffold_c:
                        p.grad.add_(scaffold_c[k] - scaffold_ci[k])
            scaler.step(opt)
            scaler.update()
            steps += 1
    return steps


@torch.no_grad()
def predict(model, bank: Bank, idx: np.ndarray, batch_size: int = 128, use_meta: bool = True) -> np.ndarray:
    model.eval()
    dev = bank.device
    out = []
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    for s in range(0, len(idx_t), batch_size):
        b = idx_t[s:s + batch_size].to(dev)
        x, m, _ = bank.batch(b, train=False)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits = model(x, m) if use_meta else model(x)
        out.append(torch.softmax(logits.float(), 1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 6), np.float32)


def balanced_accuracy(y: np.ndarray, p: np.ndarray, n_classes: int = 6) -> float:
    pred = p.argmax(1)
    rec = [np.mean(pred[y == c] == c) for c in range(n_classes) if (y == c).any()]
    return float(np.mean(rec)) if rec else float("nan")
