"""
ablate_a3 — public-release placeholder.

Inference-time component knockouts of the trained QSSM arm.

PUBLIC-RELEASE PLACEHOLDER. The interface below matches the full implementation;
the bodies are withheld until publication (see WITHHELD.md at the repository root).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qssm._withheld import WithheldError  # noqa: E402


def score(model, subs, order=None):
    raise WithheldError("score")


def fresh():
    raise WithheldError("fresh")


def main():
    raise WithheldError("main")


if __name__ == "__main__":
    sys.exit(str(WithheldError(os.path.basename(__file__))))
