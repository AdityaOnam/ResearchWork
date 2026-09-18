#!/usr/bin/env python3
"""
decoder.py

Multi-scale dual-output decoder for the RL iterative refinement model.

Reuses DSConvBlock, SEBlock, and DecoderRefine patterns from
the RDT-Net baseline decoder, extended with:
  ✅ Multi-iteration feature fusion (concatenate features from all K outer iterations)
  ✅ Dual output heads (FW and ICVF, each with independent sigmoid)
  ✅ Intermediate prediction capability for deep supervision
  ✅ Attention-weighted iteration fusion for K_outer=8
  ✅ Multi-scale feature fusion from ViT skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Building blocks (mirrored from RDT-Net for self-containedness)
# ──────────────────────────────────────────────────────────────────────
class DSConvBlock(nn.Module):
    """Depthwise-separable convolution residual block."""

    def __init__(self, channels: int, expansion: float = 1.0):
        super().__init__()
        mid = max(1, int(channels * expansion))
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1,
                      groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.net(x) + x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels // reduction),
                              bias=False)
        self.fc2 = nn.Linear(max(1, channels // reduction), channels,
                              bias=False)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))
        s = self.sig(self.fc2(self.act(self.fc1(s))))
        return x * s.view(b, c, 1, 1)


class ChannelGate(nn.Module):
    """Dual-stream channel gating (from the RDT-Net ConvGlobalSkipModel)."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, max(1, channels // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // 2), channels),
            nn.Sigmoid(),
        )

    def forward(self, a, b):
        sa, sb = a.mean(dim=(2, 3)), b.mean(dim=(2, 3))
        g = self.fc(torch.cat([sa, sb], dim=1)).unsqueeze(-1).unsqueeze(-1)
        return g * a + (1.0 - g) * b


# ──────────────────────────────────────────────────────────────────────
# Sequence-Agnostic Iteration Attention Module — dynamic K fusion
# ──────────────────────────────────────────────────────────────────────
class SequenceAgnosticIterationAttention(nn.Module):
    """Learns attention weights over a variable number of iterations (K).

    Scores each iteration independently based on its own globally pooled features,
    then applies a softmax across the temporal/iteration dimension. This allows
    fusion of any sequence length (K=1 to K=50) without hardcoded linear weights.

    Parameters
    ----------
    base_ch : int
        Feature channels per iteration.
    """

    def __init__(self, base_ch: int):
        super().__init__()
        self.base_ch = base_ch

        # Score network: evaluates the "usefulness" of a single iteration's features
        self.score_net = nn.Sequential(
            nn.Linear(base_ch, base_ch // 2),
            nn.LayerNorm(base_ch // 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_ch // 2, 1)
        )

    def forward(self, features_list: list, lengths: torch.Tensor = None) -> torch.Tensor:
        """Attention-weighted fusion of K feature maps.

        Args:
            features_list: list of dynamic length K. Each tensor is (B, base_ch, H, W).

        Returns:
            (B, base_ch, H, W) — weighted combination.
        """
        B = features_list[0].shape[0]
        K = len(features_list)

        if K == 1:
            return features_list[0]

        # Stack into (B, K, C, H, W)
        stacked = torch.stack(features_list, dim=1)

        # Global average pool each iteration -> (B, K, C)
        pooled = stacked.mean(dim=(3, 4))

        # Flatten the batch and iteration dimensions to process each independently
        pooled_flat = pooled.view(B * K, self.base_ch)

        # Compute scores and reshape -> (B, K)
        scores = self.score_net(pooled_flat).view(B, K)

        if lengths is not None:
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.tensor(lengths, dtype=torch.long, device=scores.device)
            else:
                lengths = lengths.to(device=scores.device, dtype=torch.long)
            # Create a mask of shape (B, K)
            # mask[i, j] is True if j < lengths[i], False otherwise
            arange = torch.arange(K, device=scores.device).expand(B, K)
            lengths_expanded = lengths.unsqueeze(1).expand(B, K)
            mask = arange < lengths_expanded
            
            # Apply mask by setting invalid positions to -inf before softmax
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax over the iteration dimension to get probabilities
        weights = torch.softmax(scores, dim=1)

        # Weighted sum: broadcast weights (B, K, 1, 1, 1) and sum over K
        weights = weights.view(B, K, 1, 1, 1)
        out = (stacked * weights).sum(dim=1)

        return out


# ──────────────────────────────────────────────────────────────────────
# Intermediate prediction head (used for deep supervision)
# ──────────────────────────────────────────────────────────────────────
class IntermediateHead(nn.Module):
    """Lightweight prediction head for a single refinement iteration."""

    def __init__(self, base_ch: int = 128):
        super().__init__()
        self.fw_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.icvf_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        """(B, base_ch, H, W) → (B, 1, H, W), (B, 1, H, W)."""
        return self.fw_head(features), self.icvf_head(features)


# ──────────────────────────────────────────────────────────────────────
# Main decoder: multi-scale, multi-iteration, dual-head
# ──────────────────────────────────────────────────────────────────────
class MultiScaleDecoder(nn.Module):
    """Decoder that fuses features from all K outer refinement iterations
    and produces dual FW + ICVF output maps.

    Uses attention-weighted iteration fusion + ViT skip connection
    fusion for multi-scale feature aggregation.

    Parameters
    ----------
    base_ch : int
        Feature channels per iteration (= ViT embed_dim, default 128).
    K_outer : int
        Number of outer refinement iterations (default 8).
    n_refine : int
        Number of DSConvBlocks in the refinement stack (default 3).
    se_reduction : int
        SE block channel reduction factor (default 8).
    """

    def __init__(self,
                 base_ch: int = 128,
                 n_refine: int = 3,
                 se_reduction: int = 8):
        super().__init__()
        self.base_ch = base_ch

        # Replaced the fixed IterationAttention with the Sequence-Agnostic version
        self.iter_attn = SequenceAgnosticIterationAttention(base_ch)

        # Fuse attention-weighted iterations + ViT shallow features (x0)
        fuse_in = base_ch * 2  # attention_output + x0
        self.reduce = nn.Sequential(
            nn.Conv2d(fuse_in, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

        # Multi-scale skip fusion (for ViT skip features)
        # Accepts up to 3 skip features + the reduced fused features
        self.skip_fuse = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

        # Refinement stack
        self.refines = nn.Sequential(
            *[DSConvBlock(base_ch) for _ in range(n_refine)]
        )

        # Channel attention
        self.se = SEBlock(base_ch, reduction=se_reduction)

        # Dual output heads
        self.fw_head = nn.Sequential(
            nn.Conv2d(base_ch, max(1, base_ch // 2), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, base_ch // 2), 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.icvf_head = nn.Sequential(
            nn.Conv2d(base_ch, max(1, base_ch // 2), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, base_ch // 2), 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Intermediate prediction heads (for deep supervision)
        self.intermediate_head = IntermediateHead(base_ch)

    def forward(self, all_features: list,
                x0: torch.Tensor,
                skip_features: list = None,
                lengths: torch.Tensor = None) -> tuple:
        """Final prediction from all iterations' features.

        Args:
            all_features: list of K tensors, each (B, base_ch, H, W).
            x0: (B, base_ch, H, W) — initial ViT shallow features.
            skip_features: optional list of (B, base_ch, H, W) multi-scale
                           skip features from ViT. Up to 3 features.
            lengths: optional (B,) tensor for masked iteration attention.

        Returns:
            fw_map:   (B, 1, H, W)
            icvf_map: (B, 1, H, W)
        """
        # Attention-weighted iteration fusion
        attn_fused = self.iter_attn(all_features, lengths=lengths)  # (B, base_ch, H, W)

        # Concatenate with x0 shallow features
        cat = torch.cat([attn_fused, x0], dim=1)  # (B, base_ch*2, H, W)
        x = self.reduce(cat)  # (B, base_ch, H, W)

        # Multi-scale skip fusion if available
        if skip_features and len(skip_features) > 0:
            # Pad to exactly 3 skip features if fewer
            skips = list(skip_features)
            while len(skips) < 3:
                skips.append(torch.zeros_like(x))
            skip_cat = torch.cat([x] + skips[:3], dim=1)  # (B, base_ch*4, H, W)
            x = self.skip_fuse(skip_cat)

        x = self.refines(x)
        x = self.se(x)

        fw = self.fw_head(x)
        icvf = self.icvf_head(x)
        return fw, icvf

    def intermediate(self, features: torch.Tensor) -> tuple:
        """Quick prediction from a single iteration's features.

        Used for deep supervision during training.

        Args:
            features: (B, base_ch, H, W)

        Returns:
            fw_k:   (B, 1, H, W)
            icvf_k: (B, 1, H, W)
        """
        return self.intermediate_head(features)
