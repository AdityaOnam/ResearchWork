"""
fa_md_stats — public-release placeholder.

Statistics and table generation for the downstream FA/MD results.

PUBLIC-RELEASE PLACEHOLDER. The interface below matches the full implementation;
the bodies are withheld until publication (see WITHHELD.md at the repository root).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qssm._withheld import WithheldError  # noqa: E402


def script_mae(path):
    raise WithheldError("script_mae")


def per_subject(path):
    """dMD is read from MD_KEY: `dMD` (raw) or `dMD_c` (predicted MD clipped to 3e-3)."""
    raise WithheldError("per_subject")


def holm(ps):
    raise WithheldError("holm")


def build(rows, fa_md_csv):
    raise WithheldError("build")


def contrast(a, b):
    raise WithheldError("contrast")


def latex_rows(res):
    raise WithheldError("latex_rows")


def main():
    raise WithheldError("main")


if __name__ == "__main__":
    sys.exit(str(WithheldError(os.path.basename(__file__))))
