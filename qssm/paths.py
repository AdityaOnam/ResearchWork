"""
paths — the one place the QSSM code learns where data, checkpoints and outputs live.

Nothing in this repository ships data. Every script resolves its inputs through the
environment variables below (or a `.env`-style export in your shell), so the code is
identical on every machine and no absolute path is ever hard-coded.

    QSSM_DATA_ROOT     root of your local dataset             (default: <repo>/data)
    QSSM_TRAIN_SIGNAL  training DWI subjects                  (default: $QSSM_DATA_ROOT/training)
    QSSM_TRAIN_GT      training free-water ground truth       (default: $QSSM_DATA_ROOT/FWE)
    QSSM_TEST_ROOT     held-out test subjects                 (default: $QSSM_DATA_ROOT/test)
    QSSM_PRIOR_ROOT    Stage-I ANN free-water prior maps      (default: $QSSM_DATA_ROOT/ann_prior)
    QSSM_EXT_ROOT      optional external / clinical cohort    (default: $QSSM_DATA_ROOT/external)
    QSSM_SPLITS_DIR    subject-ID lists, one ID per line      (default: <repo>/splits)
    QSSM_CKPT_DIR      model checkpoints                      (default: <repo>/checkpoints)
    QSSM_RESULTS_DIR   logs, metrics, evaluation outputs      (default: <repo>/results)
    QSSM_CACHE_DIR     synthetic-brain f_syn caches           (default: <repo>/cache)

Importing this module also puts `qssm/models`, `qssm/data` and `qssm/evaluate` on
`sys.path`, so every script can use the same flat imports (`from rdt_net import ...`).
See `docs/data_layout.md` for the directory layout each variable must point at.
"""

from __future__ import annotations

import os
import sys

QSSM = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(QSSM)

for _sub in ("models", "data", "evaluate"):
    _p = os.path.join(QSSM, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _env(name: str, default: str) -> str:
    return os.path.abspath(os.path.expanduser(os.environ.get(name, default)))


DATA_ROOT = _env("QSSM_DATA_ROOT", os.path.join(REPO, "data"))
TRAIN_SIGNAL = _env("QSSM_TRAIN_SIGNAL", os.path.join(DATA_ROOT, "training"))
TRAIN_GT = _env("QSSM_TRAIN_GT", os.path.join(DATA_ROOT, "FWE"))
TEST_ROOT = _env("QSSM_TEST_ROOT", os.path.join(DATA_ROOT, "test"))
PRIOR_ROOT = _env("QSSM_PRIOR_ROOT", os.path.join(DATA_ROOT, "ann_prior"))
EXT_ROOT = _env("QSSM_EXT_ROOT", os.path.join(DATA_ROOT, "external"))
SPLITS_DIR = _env("QSSM_SPLITS_DIR", os.path.join(REPO, "splits"))
CKPT_DIR = _env("QSSM_CKPT_DIR", os.path.join(REPO, "checkpoints"))
RESULTS_DIR = _env("QSSM_RESULTS_DIR", os.path.join(REPO, "results"))
CACHE_DIR = _env("QSSM_CACHE_DIR", os.path.join(REPO, "cache"))

# Checkpoint file names written by train/train_mamba.py, one per arm of the factorial.
ARM_CKPTS = {
    "A0": "VIT_pde_K5_Rxn_diff_synth_vit_r70_guided.pth",
    "A1": "VITQM_pde_K5_Rxn_diff_synth_hybrid_r70_guided.pth",
    "A2": "QSM_pde_K5_Rxn_diff_synth_mamba_r70_guided_noqs.pth",
    "A3": "QSM_pde_K5_Rxn_diff_synth_mamba_r70_guided.pth",
}
# The RDT-Net (ViT) checkpoint every arm warm-starts its decoder from.
WARM_START = _env("QSSM_WARM_START",
                  os.path.join(CKPT_DIR, "VIT_pde_K5_Rxn_diff_synth_r70_guided.pth"))


def ckpt(name: str) -> str:
    """Absolute path of a checkpoint file inside $QSSM_CKPT_DIR."""
    return os.path.join(CKPT_DIR, name)


def results(*parts: str) -> str:
    """Absolute path inside $QSSM_RESULTS_DIR."""
    return os.path.join(RESULTS_DIR, *parts)


def subjects(split: str) -> tuple[str, ...]:
    """Subject IDs for `split` ("train", "test", "external"), read from
    $QSSM_SPLITS_DIR/<split>.txt — one ID per line, `#` comments allowed.

    Split files are not part of the repository; see `splits/README.md`.
    """
    path = os.path.join(SPLITS_DIR, f"{split}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no subject list at {path}. Create it (one subject ID per line) or point "
            f"QSSM_SPLITS_DIR at a directory that has one; see splits/README.md.")
    with open(path) as fh:
        ids = [ln.split("#", 1)[0].strip() for ln in fh]
    return tuple(i for i in ids if i)
