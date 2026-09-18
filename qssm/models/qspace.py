"""
qspace — public-release placeholder.

Canonical geodesic q-space traversal of the gradient directions.

PUBLIC-RELEASE PLACEHOLDER. The interface below matches the full implementation;
the bodies are withheld until publication (see WITHHELD.md at the repository root).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qssm._withheld import WithheldError  # noqa: E402

__all__ = ['reference_scheme', 'canonical_order', 'order_table', 'random_order', 'identity_order', 'TRAIN_SIGNAL']


def reference_scheme(subjects=None):
    """Mean gradient scheme over the cohort. Returns `(bvecs (3,91), bvals (91,))`."""
    raise WithheldError("reference_scheme")


def canonical_order(subjects=None):
    """The fixed 91-element permutation every arm scans in, unless ablated."""
    raise WithheldError("canonical_order")


def order_table(subjects):
    """`(n_subjects, 91)` -- each subject's own greedy order. Per-subject ablation only."""
    raise WithheldError("order_table")


def random_order(bvals=None, seed=7):
    """A fixed random permutation, b0 pinned first."""
    raise WithheldError("random_order")


def identity_order(n=91):
    """File order -- the scan with no angular structure at all."""
    raise WithheldError("identity_order")


def check(verbose=True):
    """Recompute the canonical order and compare against the recorded permutation."""
    raise WithheldError("check")


if __name__ == "__main__":
    sys.exit(str(WithheldError(os.path.basename(__file__))))
