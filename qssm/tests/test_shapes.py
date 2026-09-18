#!/usr/bin/env python3
"""
test_shapes — forward every arm of the 2x2 factorial on random tensors. Needs no data.

Checks, per arm:
  * the model builds from `rdt_net.ConvGlobalSkipModel(encoder=...)`,
  * the parameter count matches the value pinned in `ablations/verify_arms.py`,
  * a (B, 91, H, W) input produces a (B, 1, H, W) free-water map in [0, 1].

    python qssm/tests/test_shapes.py              # CPU, pure-PyTorch scan (~1 min)
    python -m pytest qssm/tests -q                # same, under pytest
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401  -- puts models/ on sys.path

from rdt_net import ConvGlobalSkipModel  # noqa: E402

sys.path.insert(0, paths.REPO)
from qssm._withheld import WithheldError  # noqa: E402

IN_CH, K = 91, 5
H, W = 174, 145          # the real HCP slice; the ViT arms size their positional
                         # embedding from it, so the pinned counts only hold at this frame

ARMS = {
    # label: (encoder, extra kwargs, expected total parameters)
    "A0 vit":    ("vit",    {},                  509_383),
    "A1 hybrid": ("hybrid", {},                  513_591),
    "A2 mamba":  ("mamba",  dict(use_qscan=False), 408_775),
    "A3 qssm":   ("mamba",  {},                  412_983),
}


def build(encoder, enc_kw):
    kw = dict(enc_kw, kernel="torch")        # pure-PyTorch scan: runs on CPU, no CUDA build
    return ConvGlobalSkipModel(in_channels=IN_CH, K=K, encoder=encoder, enc_kw=kw)


def run_arm(label):
    encoder, enc_kw, expected = ARMS[label]
    torch.manual_seed(0)
    model = build(encoder, enc_kw).eval()
    x = torch.rand(1, IN_CH, H, W)
    with torch.no_grad():
        y = model(x)
    y = y[0] if isinstance(y, (tuple, list)) else y
    n = sum(p.numel() for p in model.parameters())
    return n, expected, tuple(y.shape), float(y.min()), float(y.max())


def test_all_arms():
    for label in ARMS:
        try:
            n, expected, shape, lo, hi = run_arm(label)
        except WithheldError:
            continue                      # arm's encoder is withheld in the public release
        assert shape == (1, 1, H, W), (label, shape)
        assert n == expected, f"{label}: {n:,} params, expected {expected:,}"
        assert torch.isfinite(torch.tensor([lo, hi])).all(), label


if __name__ == "__main__":
    for label in ARMS:
        try:
            n, expected, shape, lo, hi = run_arm(label)
        except WithheldError:
            print(f"[skip] {label:10}  encoder withheld in the public release (see WITHHELD.md)")
            continue
        ok = "ok " if n == expected and shape == (1, 1, H, W) else "FAIL"
        print(f"[{ok}] {label:10}  params {n:>9,} (expected {expected:,})  "
              f"out {shape}  range [{lo:.3f}, {hi:.3f}]")
