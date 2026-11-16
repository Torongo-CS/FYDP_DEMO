"""Image-only and image+metadata models on a MobileNetV2 backbone.

fusion = "image"  : image features only
fusion = "concat" : image features and metadata features side by side
fusion = "gated"  : a learned gate mixes the two (simplified from Fathallah et al. 2026)
fusion = "xattn"  : metadata tokens look at the 7x7 image feature map
                    (cross-attention, as in Ou et al. 2022 / Mridha & Islam 2026)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

FUSIONS = ["image", "concat", "gated", "xattn"]


def mobilenet_v2_features(pretrained: bool, weights_path: str | None = None) -> nn.Module:
    m = torchvision.models.mobilenet_v2(weights=None)
    if pretrained:
        if weights_path:
            m.load_state_dict(torch.load(weights_path, map_location="cpu"))
        else:
            w = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
            m = torchvision.models.mobilenet_v2(weights=w)
    return m.features  # output: (B, 1280, H/32, W/32)


class MetaMLP(nn.Module):
    def __init__(self, d_in: int, d: int = 128, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d), nn.LayerNorm(d), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d, d), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class SkinNet(nn.Module):
    def __init__(self, fusion: str, meta_dim: int, n_classes: int = 6, pretrained: bool = True,
                 weights_path: str | None = None, d: int = 256, p: float = 0.2):
        super().__init__()
        assert fusion in FUSIONS, fusion
        self.fusion = fusion
        self.backbone = mobilenet_v2_features(pretrained, weights_path)
        c = 1280
        self.drop = nn.Dropout(p)
        if fusion == "image":
            self.head = nn.Linear(c, n_classes)
            return
        self.meta = MetaMLP(meta_dim, 128, p)
        self.img_proj = nn.Sequential(nn.Linear(c, d), nn.ReLU())
        if fusion == "concat":
            self.head = nn.Linear(d + 128, n_classes)
        elif fusion == "gated":
            self.meta_proj = nn.Linear(128, d)
            self.gate = nn.Linear(d + d, d)
            self.head = nn.Linear(d, n_classes)
        elif fusion == "xattn":
            self.n_tok = 4
            self.tok = nn.Linear(128, self.n_tok * d)
            self.kv = nn.Conv2d(c, d, 1)
            self.pos = nn.Parameter(torch.zeros(1, 64, d))
            self.attn = nn.MultiheadAttention(d, num_heads=4, dropout=p, batch_first=True)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(2 * d, n_classes)

    def forward(self, x, meta=None, return_attn: bool = False):
        fmap = self.backbone(x)
        g = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        if self.fusion == "image":
            return self.head(self.drop(g))
        m = self.meta(meta)
        gi = self.img_proj(g)
        if self.fusion == "concat":
            return self.head(self.drop(torch.cat([gi, m], 1)))
        if self.fusion == "gated":
            mp = self.meta_proj(m)
            a = torch.sigmoid(self.gate(torch.cat([gi, mp], 1)))
            return self.head(self.drop(a * gi + (1 - a) * mp))
        # cross-attention
        kv = self.kv(fmap).flatten(2).transpose(1, 2)  # (B, HW, d)
        kv = kv + self.pos[:, : kv.shape[1]]
        q = self.tok(m).view(m.shape[0], self.n_tok, -1)
        out, w = self.attn(q, kv, kv, need_weights=return_attn)
        z = self.norm(out.mean(1))
        logits = self.head(self.drop(torch.cat([gi, z], 1)))
        return (logits, w) if return_attn else logits


def bn_keys(model: nn.Module) -> set[str]:
    mods = dict(model.named_modules())
    keys = set()
    for k in model.state_dict():
        m = mods.get(k.rsplit(".", 1)[0])
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            keys.add(k)
    return keys


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
