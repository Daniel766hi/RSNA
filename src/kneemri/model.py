"""Position-aware 2.5-D attention-MIL with per-finding queries and label interaction.

    windows [B,W,3,H,W]
      -> 2-D backbone (timm, any)                         [B,W,F]
      -> linear projection + slot embedding + position encoding (lever L2)
      -> window-context transformer (TransMIL-style)      [B,W,d]
      -> 12 finding queries cross-attend to windows       [B,12,d]   (Query2Label / ML-Decoder)
      -> label-interaction self-attention over findings   [B,12,d]   (lever L3)
      -> per-finding linear logit                         [B,12]

Why not plain attention-MIL? The public CoAtNets pool windows with a permutation-invariant
attention, so they cannot tell where along the slice axis a window lies. On sagittal slots
(after mirroring right knees) that position separates the medial from the lateral compartment,
which is exactly what Medial vs Lateral Meniscus / OA require. The position encoding is
slot-conditioned, because position means medial-lateral on sagittal, anterior-posterior on
coronal and superior-inferior on axial slots.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .constants import N_TARGETS


@dataclass
class ModelConfig:
    backbone: str = "convnext_small.fb_in22k_ft_in1k_384"
    pretrained: bool = True
    n_slots: int = 5
    d_model: int = 512
    n_heads: int = 8
    ctx_layers: int = 1
    label_layers: int = 1
    dropout: float = 0.1
    n_freqs: int = 8
    use_position: bool = True
    grad_ckpt: bool = False
    backbone_chunk: int = 0     # >0: run the backbone over windows in chunks (inference memory)

    def to_dict(self) -> dict:
        return asdict(self)


def fourier(pos: torch.Tensor, n_freqs: int) -> torch.Tensor:
    freqs = (2.0 ** torch.arange(n_freqs, device=pos.device, dtype=pos.dtype)) * math.pi
    ang = pos[..., None] * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class KneeMIL(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None, n_targets: int = N_TARGETS):
        super().__init__()
        import timm  # noqa: PLC0415

        self.cfg = cfg = cfg or ModelConfig()
        self.backbone = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=0, in_chans=3)
        if cfg.grad_ckpt and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        f = self.backbone.num_features
        d = cfg.d_model
        self.proj = nn.Sequential(nn.Linear(f, d), nn.LayerNorm(d))
        self.slot_emb = nn.Embedding(cfg.n_slots, d)
        self.pos_mlp = nn.Sequential(nn.Linear(2 * cfg.n_freqs + cfg.n_slots, d), nn.GELU(), nn.Linear(d, d))
        layer = lambda: nn.TransformerEncoderLayer(  # noqa: E731
            d, cfg.n_heads, 2 * d, cfg.dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.ctx = nn.ModuleList([layer() for _ in range(cfg.ctx_layers)])
        self.queries = nn.Parameter(torch.randn(n_targets, d) * 0.02)
        self.cross = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d)
        self.label_mix = nn.ModuleList([layer() for _ in range(cfg.label_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.head_w = nn.Parameter(torch.randn(n_targets, d) * 0.02)
        self.head_b = nn.Parameter(torch.zeros(n_targets))

    def encode_windows(self, x: torch.Tensor) -> torch.Tensor:
        """[N,3,H,W] -> [N,F], optionally in chunks to bound activation memory."""
        c = self.cfg.backbone_chunk
        if c and x.shape[0] > c:
            return torch.cat([self.backbone(x[i: i + c]) for i in range(0, x.shape[0], c)])
        return self.backbone(x)

    def forward(self, x: torch.Tensor, slot: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, w = mask.shape
        mask = mask.clone()
        mask[~mask.any(1), 0] = True          # a study with no windows attends to one zero window
        feats = torch.zeros(b, w, self.backbone.num_features, device=x.device, dtype=x.dtype)
        valid = mask.reshape(-1)
        enc = self.encode_windows(x.reshape(b * w, *x.shape[2:])[valid])
        feats = feats.reshape(b * w, -1)
        feats[valid] = enc.to(feats.dtype)
        h = self.proj(feats.reshape(b, w, -1).float())
        h = h + self.slot_emb(slot)
        if self.cfg.use_position:
            onehot = torch.nn.functional.one_hot(slot, self.cfg.n_slots).float()
            h = h + self.pos_mlp(torch.cat([fourier(pos.float(), self.cfg.n_freqs), onehot], dim=-1))
        pad = ~mask
        for blk in self.ctx:
            h = cp.checkpoint(blk, h, None, pad, use_reentrant=False) if (self.cfg.grad_ckpt and self.training) \
                else blk(h, src_key_padding_mask=pad)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        z, _ = self.cross(q, h, h, key_padding_mask=pad, need_weights=False)
        z = self.cross_norm(z + q)
        for blk in self.label_mix:
            z = blk(z)
        z = self.out_norm(z)
        return (z * self.head_w).sum(-1) + self.head_b


def build_model(cfg: ModelConfig) -> KneeMIL:
    return KneeMIL(cfg)


def param_groups(model: KneeMIL, lr_backbone: float, lr_head: float, wd: float) -> list[dict]:
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    return [
        {"params": bb, "lr": lr_backbone, "weight_decay": wd},
        {"params": head, "lr": lr_head, "weight_decay": wd},
    ]
