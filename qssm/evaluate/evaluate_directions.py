"""
evaluate_directions — public-release placeholder.

Robustness sweep under gradient-direction subsampling.

PUBLIC-RELEASE PLACEHOLDER. The interface below matches the full implementation;
the bodies are withheld until publication (see WITHHELD.md at the repository root).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qssm._withheld import WithheldError  # noqa: E402


def load_normalized(dwi_path):
    """The load+normalise half of prepare_dwi_b0_flat_and_slices, done once per subject.
    Arithmetic copied verbatim (b0 clip, divide, nan/negative/1.0 clamp) so this is
    provably the same normalisation, just cached instead of recomputed per level."""
    raise WithheldError("load_normalized")


def select_and_pad(data2d_norm, dims, keep_nonb0):
    """The selection+reshape half: angularly-uniform b0-plus-linspace indices, zero-padded
    to the full channel count, transposed to (S, C, H, W) -- matches
    prepare_dwi_b0_flat_and_slices's tail exactly."""
    raise WithheldError("select_and_pad")


def score_all_levels(model, sid, keep_levels, mask_input=True):
    """One subject, ALL direction-counts, one disk read. Returns {keep_nonb0: (mae, n_kept)}."""
    raise WithheldError("score_all_levels")


def main():
    raise WithheldError("main")


if __name__ == "__main__":
    sys.exit(str(WithheldError(os.path.basename(__file__))))
