"""
ablate_gate_se — public-release placeholder.

Saturation probes and knockouts of ChannelGate / SEBlock.

PUBLIC-RELEASE PLACEHOLDER. The interface below matches the full implementation;
the bodies are withheld until publication (see WITHHELD.md at the repository root).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qssm._withheld import WithheldError  # noqa: E402


def attach_probes(model):
    """Record the gate/SE activations each block emits, per forward."""
    raise WithheldError("attach_probes")


def knock_out(model, gate=None, se=False):
    """Patch blocks in place. `gate` in {None,'a','mean'}; `se=True` -> identity."""
    raise WithheldError("knock_out")


def freeze_to_constant(model, rec):
    """Replace each block's sigmoid output with its dataset-mean vector."""
    raise WithheldError("freeze_to_constant")


def score(model, in_ch, subjects):
    raise WithheldError("score")


def main():
    raise WithheldError("main")


if __name__ == "__main__":
    sys.exit(str(WithheldError(os.path.basename(__file__))))
