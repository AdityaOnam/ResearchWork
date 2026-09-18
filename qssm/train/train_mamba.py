"""
train_mamba — public-release placeholder.

Trainer for the four arms of the 2x2 encoder factorial.

PUBLIC-RELEASE PLACEHOLDER. The interface below matches the full implementation;
the bodies are withheld until publication (see WITHHELD.md at the repository root).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qssm._withheld import WithheldError  # noqa: E402


class Logger:

    def __init__(self, out_dir):
        raise WithheldError("Logger.__init__")

    def row(self, *vals):
        raise WithheldError("Logger.row")

    def best(self, payload):
        raise WithheldError("Logger.best")


def validate(model, loader, order_table=None):
    """Real slices only. RDT-Net's MAE: `mean(|p*m - y*m|)` over the full frame."""
    raise WithheldError("validate")


def build_args():
    raise WithheldError("build_args")


def run_tag(args):
    raise WithheldError("run_tag")


def main():
    raise WithheldError("main")


if __name__ == "__main__":
    sys.exit(str(WithheldError(os.path.basename(__file__))))
