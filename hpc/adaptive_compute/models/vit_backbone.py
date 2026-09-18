#!/usr/bin/env python3
"""
vit_backbone.py

Vision Transformer backbone for global anatomical feature learning.

Reused from the RDT-Net baseline (rdtnet_baseline/train_rxn_diff.py, SimpleViT),
extended with:
  ✅ Multi-scale feature hooks for the iterative decoder
  ✅ Multi-scale skip outputs at configurable depths
  ✅ Supports increased capacity (embed_dim=128, depth=8, heads=16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderBlock(nn.Module):
    """Standard Pre-LN Transformer encoder block with MHSA + MLP."""

    def __init__(self, dim: int, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            batch_first=True, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x_n = self.norm1(x)
        x = x + self.attn(x_n, x_n, x_n)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleViT(nn.Module):
    """Lightweight ViT for 2-D DWI slices.

    Pipeline:
        Conv patch embedding  →  positional encoding  →  Transformer blocks
        →  LayerNorm  →  reshape to 2-D  →  bilinear upsample to input H×W

    The ``forward`` method returns the final feature map.  Call
    ``forward_multiscale`` to also receive intermediate features
    (useful for deep supervision in the iterative decoder).

    Parameters
    ----------
    in_ch : int
        Input channel count (93 = 91 DWI + 1 FW prior + 1 ICVF prior).
    embed_dim : int
        Transformer embedding dimension (default 128).
    patch_size : int
        Spatial patch size for the conv embedding (default 4).
    depth : int
        Number of Transformer encoder blocks (default 8).
    num_heads : int
        Multi-head attention heads (default 16).
    mlp_ratio : float
        MLP hidden-dim expansion ratio (default 4.0).
    skip_indices : list[int] or None
        Block indices (0-based) from which to extract skip features
        for multi-scale fusion. If None, uses [depth//4, depth//2, 3*depth//4].
    """

    def __init__(self, in_ch: int = 93, embed_dim: int = 128,
                 patch_size: int = 4, depth: int = 8,
                 num_heads: int = 16, mlp_ratio: float = 4.0,
                 skip_indices: list = None):
        super().__init__()
        assert patch_size >= 1 and isinstance(patch_size, int)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth

        # Multi-scale skip connection indices
        if skip_indices is None:
            self.skip_indices = [
                max(0, depth // 4 - 1),
                max(0, depth // 2 - 1),
                max(0, 3 * depth // 4 - 1),
            ]
        else:
            self.skip_indices = skip_indices

        # Patch embedding via strided convolution
        self.patch_embed = nn.Conv2d(
            in_ch, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio
            )
            for _ in range(depth)
        ])

        self.pos_emb = None  # lazily initialised on first forward pass
        self.norm = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------
    def _init_pos_emb(self, N: int, device: torch.device):
        """Lazily create or resize positional embedding."""
        if self.pos_emb is None or self.pos_emb.shape[1] != N:
            pe = torch.zeros(1, N, self.embed_dim, device=device)
            nn.init.trunc_normal_(pe, std=0.02)
            self.pos_emb = nn.Parameter(pe)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward: returns final upsampled feature map.

        Args:
            x: (B, in_ch, H, W)

        Returns:
            (B, embed_dim, H, W) — feature map at input resolution.
        """
        B, C, H, W = x.shape

        # Patch embed
        x_p = self.patch_embed(x)                      # (B, D, Hp, Wp)
        Hp, Wp = x_p.shape[2], x_p.shape[3]
        N = Hp * Wp

        tokens = x_p.flatten(2).transpose(1, 2)        # (B, N, D)

        # Positional encoding
        self._init_pos_emb(N, x.device)
        tokens = tokens + self.pos_emb

        # Transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.norm(tokens)

        # Reshape back to 2-D and upsample
        tokens = tokens.transpose(1, 2).view(B, self.embed_dim, Hp, Wp)
        out = F.interpolate(tokens, size=(H, W),
                            mode="bilinear", align_corners=False)
        return out

    # ------------------------------------------------------------------
    def forward_with_skips(self, x: torch.Tensor) -> tuple:
        """Forward pass returning final features AND multi-scale skip features.

        Used for stronger residual skip connections in the iterative model.

        Args:
            x: (B, in_ch, H, W)

        Returns:
            final : (B, D, H, W) — upsampled final feature map.
            skip_features : list[(B, D, H, W)] — features at skip_indices.
        """
        B, C, H, W = x.shape

        x_p = self.patch_embed(x)
        Hp, Wp = x_p.shape[2], x_p.shape[3]
        N = Hp * Wp

        tokens = x_p.flatten(2).transpose(1, 2)
        self._init_pos_emb(N, x.device)
        tokens = tokens + self.pos_emb

        skip_features = []
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if i in self.skip_indices:
                feat = tokens.transpose(1, 2).view(B, self.embed_dim, Hp, Wp)
                feat_up = F.interpolate(feat, size=(H, W),
                                        mode="bilinear", align_corners=False)
                skip_features.append(feat_up)

        tokens = self.norm(tokens)
        final = tokens.transpose(1, 2).view(B, self.embed_dim, Hp, Wp)
        final = F.interpolate(final, size=(H, W),
                              mode="bilinear", align_corners=False)

        return final, skip_features

    # ------------------------------------------------------------------
    def forward_multiscale(self, x: torch.Tensor) -> tuple:
        """Forward pass that also returns intermediate features.

        Returns:
            final : (B, D, H, W) — upsampled final feature map.
            intermediates : list[(B, D, H, W)] — one per block.
        """
        B, C, H, W = x.shape

        x_p = self.patch_embed(x)
        Hp, Wp = x_p.shape[2], x_p.shape[3]
        N = Hp * Wp

        tokens = x_p.flatten(2).transpose(1, 2)
        self._init_pos_emb(N, x.device)
        tokens = tokens + self.pos_emb

        intermediates = []
        for blk in self.blocks:
            tokens = blk(tokens)
            # Reshape intermediate features to spatial
            feat = tokens.transpose(1, 2).view(B, self.embed_dim, Hp, Wp)
            feat_up = F.interpolate(feat, size=(H, W),
                                    mode="bilinear", align_corners=False)
            intermediates.append(feat_up)

        tokens = self.norm(tokens)
        final = tokens.transpose(1, 2).view(B, self.embed_dim, Hp, Wp)
        final = F.interpolate(final, size=(H, W),
                              mode="bilinear", align_corners=False)

        return final, intermediates
