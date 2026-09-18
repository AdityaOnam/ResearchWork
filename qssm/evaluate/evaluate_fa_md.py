"""
evaluate_fa_md — public-release placeholder.

Downstream FA/MD accuracy after free-water correction.

PUBLIC-RELEASE PLACEHOLDER. The interface below matches the full implementation;
the bodies are withheld until publication (see WITHHELD.md at the repository root).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qssm._withheld import WithheldError  # noqa: E402


def subject_paths(cohort, sid):
    raise WithheldError("subject_paths")


def load3d(p):
    raise WithheldError("load3d")


def design(gtab):
    """Log-linear tensor design, b in ms/um^2 so the normal equations stay well conditioned.
    Columns: xx, yy, zz, xy, xz, yz, ln S0."""
    raise WithheldError("design")


def sym3_eigvals(dxx, dyy, dzz, dxy, dxz, dyz):
    """Closed-form eigenvalues of a symmetric 3x3 (Smith 1961). torch.linalg.eigvalsh on a
    batch of 1e5 tensors asks cuSOLVER for ~50 GB of workspace; this needs none."""
    raise WithheldError("sym3_eigvals")


def tensor_fa_md(dxx, dyy, dzz, dxy, dxz, dyz):
    """FA/MD from tensor elements in mm^2/s; negative eigenvalues clipped to 0 as DIPY does."""
    raise WithheldError("tensor_fa_md")


def wls_beta(S, X, min_signal):
    """DIPY's WLS: OLS in log space, then refit weighted by the OLS-predicted signal squared."""
    raise WithheldError("wls_beta")


def corrected_fa_md(sig, f, gtab, X, Xd, min_signal, fit='wls', pool=None):
    """FA/MD of the tissue tensor once f of the signal is attributed to free water."""
    raise WithheldError("corrected_fa_md")


def minmax(v):
    raise WithheldError("minmax")


def score_subject(cohort, sid, variants=None, fit='wls', pool=None):
    raise WithheldError("score_subject")


def main():
    raise WithheldError("main")


if __name__ == "__main__":
    sys.exit(str(WithheldError(os.path.basename(__file__))))
