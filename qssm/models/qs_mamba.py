"""
qs_mamba — public-release placeholder.

QSSM encoder: selective scan (S6), spatial cross-scan blocks, bidirectional q-space scan, QSMambaEncoder.

PUBLIC-RELEASE PLACEHOLDER. The interface below matches the full implementation;
the bodies are withheld until publication (see WITHHELD.md at the repository root).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qssm._withheld import WithheldError  # noqa: E402
from torch import nn  # noqa: E402

__all__ = ['QSMambaEncoder', 'QSpaceScanBi', 'S6', 'SpatialScanBlock', 'qspace_order', 'selective_scan', 'pscan']


def qspace_order(bvecs, bvals=None, b0_thresh=50.0):
    """Greedy nearest-neighbour traversal of the gradient directions."""
    raise WithheldError("qspace_order")


def pscan(a, x):
    """Inclusive scan of `h_t = a_t * h_{t-1} + x_t` along dim 2."""
    raise WithheldError("pscan")


def selective_scan(u, delta, a_log, b, c, d, chunk=2048, checkpoint=True):
    """Mamba's S6. `u, delta: (B,D,L)`, `a_log: (D,N)`, `b, c: (B,N,L)`, `d: (D,)`."""
    raise WithheldError("selective_scan")


class S6(nn.Module):
    """One selective-scan channel mixer over a sequence axis."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None, chunk=2048, checkpoint=True, kernel='auto'):
        raise WithheldError("S6.__init__")

    def forward(self, x):
        """`x: (B, L, D)` -> `(B, L, D)`. Backend per `self.kernel`; see `ssm_ops`."""
        raise WithheldError("S6.forward")


class SpatialScanBlock(nn.Module):
    """Bidirectional (or 4-way cross) selective scan over the patch grid."""

    def __init__(self, dim, d_state=16, expand=2, n_scan_dirs=4, mlp_ratio=2.0, kernel='auto'):
        raise WithheldError("SpatialScanBlock.__init__")

    def forward(self, x):
        """`x: (B, D, Hp, Wp)`."""
        raise WithheldError("SpatialScanBlock.forward")


class QSpaceScanBi(nn.Module):
    """Bidirectional selective scan along the gradient-direction axis."""

    def __init__(self, in_ch, patch_size=4, d_model=16, d_state=8, d_out=16, expand=2, q_stride=2, kernel='auto'):
        raise WithheldError("QSpaceScanBi.__init__")

    def forward(self, x, order=None, out_hw=None):
        """`x: (B, V, H, W)` -> `(B, d_out, Hp, Wp)`."""
        raise WithheldError("QSpaceScanBi.forward")


class QSMambaEncoder(nn.Module):
    """Drop-in replacement for `SimpleViT`."""

    def __init__(self, in_ch, embed_dim=64, patch_size=4, depth=4, d_state=16, expand=2, n_scan_dirs=4, use_qscan=True, q_dim=16, q_state=8, q_stride=2, emit_dt=False, kernel='auto'):
        raise WithheldError("QSMambaEncoder.__init__")

    def set_qspace_order(self, order):
        """Install a per-subject traversal from `qspace_order(bvecs, bvals)`."""
        raise WithheldError("QSMambaEncoder.set_qspace_order")

    def forward(self, x, order=None):
        raise WithheldError("QSMambaEncoder.forward")
