#!/usr/bin/env python3
"""
encoders — the three encoder arms of the 2x2 factorial, behind one constructor.

    kind        spatial operator          q-space operator       params (encoder / total)
    ----------------------------------------------------------------------------------
    vit         attention (4 blocks)      none                   392,384 / 509,383   A0
    hybrid      attention (4 blocks)      bi-dir selective scan  396,592 / 513,591   A1
    mamba       4-way spatial cross-scan  bi-dir selective scan  295,984 / 412,983   A3
    mamba       4-way spatial cross-scan  none (--no-qscan)      291,776 / 408,775   A2

Every encoder has the same contract as `SimpleViT`: `(B, in_ch, H, W) -> (B, embed_dim,
H, W)`, bilinearly upsampled from the patch grid, and is installed as
`ConvGlobalSkipModel.in_vit`. Keeping that attribute name is not cosmetic — it is what
makes the warm start land.

Why `hybrid` gives Mamba only the q-space axis
----------------------------------------------
`QHybridViT` subclasses `SimpleViT` and adds exactly two submodules, so its state-dict keys
are `in_vit.patch_embed.*`, `in_vit.pos_emb`, `in_vit.blocks.*`, `in_vit.norm.*` —
byte-identical to the baseline — plus `in_vit.qscan.*` and `in_vit.q_merge.*`. Loading
`VIT_pde_K5_Rxn_diff_synth_r70_guided.pth` with `strict=False` fills every parameter except
the q-branch, whose output projection is zero-initialised.

**So A1 at step 0 computes the warm-start baseline bit-for-bit.** It cannot start worse than the
baseline, and the novel path has to earn every digit. That is the sharpest experimental
design available here and the whole arm is structured around it — `verify_arms.py` asserts
it before any training runs.

Interleaving attention and scan blocks Jamba-style is *not* an option for the hybrid, and
not for a boring reason: `patch_embed` consumes the 91-direction axis and never returns it,
so a q-scan block placed at ViT depth 3 would have to re-read the raw input. It would be a
parallel branch merging late, not an interleave — and merging late is strictly worse, since
the four attention blocks would then never see the q-space feature and the decoder's `x0`
skip would bypass it too. Injection happens before `pos_emb` and before block 0.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qs_mamba import QSMambaEncoder, QSpaceScanBi  # noqa: E402
from rdt_net import SimpleViT  # noqa: E402

__all__ = ["ViTEncoder", "QHybridViT", "QSMambaEncoder", "build_encoder"]


class ViTEncoder(SimpleViT):
    """The unmodified baseline, with an `order=` parameter it ignores.

    Subclassing rather than wrapping keeps every state-dict key identical to `SimpleViT`,
    so the A0 control loads the published checkpoint 1:1.
    """

    def forward(self, x, order=None):
        return super().forward(x)


class QHybridViT(SimpleViT):
    """A1 — attention over space, a bidirectional selective scan over q-space."""

    def __init__(self, in_ch, embed_dim, patch_size=4, depth=4, num_heads=8,
                 mlp_ratio=4.0, q_dim=16, q_state=8, q_stride=2, kernel="auto"):
        super().__init__(in_ch, embed_dim, patch_size=patch_size, depth=depth,
                         num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.qscan = QSpaceScanBi(in_ch, patch_size, d_model=q_dim, d_state=q_state,
                                  d_out=q_dim, q_stride=q_stride, kernel=kernel)
        self.q_merge = nn.Conv2d(q_dim, embed_dim, 1, bias=False)
        nn.init.zeros_(self.q_merge.weight)      # exact baseline at init
        # Non-persistent: the traversal is a property of the acquisition scheme, not of
        # the trained weights, and it must not enter the state dict.
        self.register_buffer("q_order", torch.arange(in_ch), persistent=False)

    def set_qspace_order(self, order):
        self.q_order = torch.as_tensor(order, dtype=torch.long,
                                       device=self.q_order.device)

    def forward(self, x, order=None):
        b, c, h, w = x.shape
        x_p = self.patch_embed(x)                                  # (B, D, Hp, Wp)

        o = order if order is not None else self.q_order
        x_p = x_p + self.q_merge(self.qscan(x, o, out_hw=x_p.shape[-2:]))

        hp, wp = x_p.shape[2], x_p.shape[3]
        n = hp * wp
        tokens = x_p.flatten(2).transpose(1, 2)                    # (B, N, D)

        # Lazy positional embedding, verbatim from SimpleViT. It is why the ViT arms need a
        # dummy forward before load_state_dict; the pure-Mamba arm has no pos_emb at all.
        if (self.pos_emb is None) or (self.pos_emb.shape[1] != n):
            pe = torch.zeros(1, n, self.embed_dim, device=x.device)
            nn.init.trunc_normal_(pe, std=0.02)
            self.pos_emb = nn.Parameter(pe)

        tokens = tokens + self.pos_emb
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        tokens = tokens.transpose(1, 2).view(b, self.embed_dim, hp, wp)
        return nn.functional.interpolate(tokens, size=(h, w), mode="bilinear",
                                         align_corners=False)


def build_encoder(kind, in_ch=91, embed_dim=64, *, patch_size=4, depth=4, num_heads=8,
                  mlp_ratio=4.0, use_qscan=True, q_dim=16, q_state=8, q_stride=2,
                  d_state=16, expand=2, n_scan_dirs=4, emit_dt=False, kernel="auto"):
    """`kind` in {"vit", "hybrid", "mamba"}. See the table in the module docstring."""
    if kind == "vit":
        return ViTEncoder(in_ch, embed_dim, patch_size=patch_size, depth=depth,
                          num_heads=num_heads, mlp_ratio=mlp_ratio)
    if kind == "hybrid":
        return QHybridViT(in_ch, embed_dim, patch_size=patch_size, depth=depth,
                          num_heads=num_heads, mlp_ratio=mlp_ratio, q_dim=q_dim,
                          q_state=q_state, q_stride=q_stride, kernel=kernel)
    if kind == "mamba":
        return QSMambaEncoder(in_ch, embed_dim=embed_dim, patch_size=patch_size,
                              depth=depth, d_state=d_state, expand=expand,
                              n_scan_dirs=n_scan_dirs, use_qscan=use_qscan, q_dim=q_dim,
                              q_state=q_state, q_stride=q_stride, emit_dt=emit_dt,
                              kernel=kernel)
    raise ValueError(f"unknown encoder {kind!r}; expected vit, hybrid or mamba")
