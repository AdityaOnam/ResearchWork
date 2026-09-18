#!/usr/bin/env python3
"""
setup_check — first-run verification. Needs no data.

Reports, in order:
  1. Python / PyTorch / CUDA versions
  2. required and optional packages (mamba-ssm, dipy, TractSeg, ...)
  3. external neuroimaging tools on PATH (MRtrix3, SS3T-CSD, FSL)
  4. the QSSM path variables and which directories exist
  5. subject split files
  6. a forward pass of every buildable arm (withheld arms are reported, not failed)

    python scripts/setup_check.py
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "
failures = 0


def row(tag, name, detail=""):
    global failures
    failures += tag == FAIL
    print(f"[{tag}] {name:<28} {detail}")


def section(title):
    print(f"\n== {title} " + "=" * (60 - len(title)))


section("runtime")
row(OK if sys.version_info >= (3, 10) else FAIL, "python", platform.python_version())
try:
    import torch
    row(OK, "torch", f"{torch.__version__}  cuda={torch.version.cuda}  "
                     f"gpu={'yes: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no'}")
except ImportError:
    row(FAIL, "torch", "not installed (pip install torch)")

section("python packages")
for mod, required in [("numpy", 1), ("scipy", 1), ("nibabel", 1), ("timm", 1), ("tqdm", 1),
                      ("dipy", 0), ("yaml", 0), ("matplotlib", 0),
                      ("mamba_ssm", 0), ("causal_conv1d", 0), ("tractseg", 0)]:
    try:
        m = importlib.import_module(mod)
        row(OK, mod, getattr(m, "__version__", ""))
    except Exception:
        row(FAIL if required else WARN, mod, "missing" + ("" if required else " (optional)"))

section("external tools (tractography / preprocessing only)")
for tool in ("mrconvert", "dwi2fod", "tckgen", "ss3t_csd_beta1", "TractSeg", "dwifslpreproc", "eddy"):
    p = shutil.which(tool)
    row(OK if p else WARN, tool, p or "not on PATH (optional)")

section("paths (qssm/paths.py)")
sys.path.insert(0, os.path.join(REPO, "qssm"))
import paths  # noqa: E402
for name in ("DATA_ROOT", "TRAIN_SIGNAL", "TRAIN_GT", "TEST_ROOT", "PRIOR_ROOT",
             "SPLITS_DIR", "CKPT_DIR", "RESULTS_DIR"):
    p = getattr(paths, name)
    row(OK if os.path.isdir(p) else WARN, name, p + ("" if os.path.isdir(p) else "  (missing)"))

section("subject splits")
for split in ("train", "test", "external"):
    try:
        row(OK, f"{split}.txt", f"{len(paths.subjects(split))} subjects")
    except FileNotFoundError:
        row(WARN, f"{split}.txt", "not created yet (see splits/README.md)")

section("model build (no data)")
r = subprocess.run([sys.executable, os.path.join(REPO, "qssm", "tests", "test_shapes.py")],
                   capture_output=True, text=True)
for line in r.stdout.splitlines():
    if line.startswith("["):
        tag = OK if line.startswith("[ok") else WARN if line.startswith("[skip") else FAIL
        tok = line.split("]", 1)[1].split()
        row(tag, " ".join(tok[:2]), " ".join(tok[2:]))
if r.returncode:
    row(FAIL, "test_shapes.py", r.stderr.strip().splitlines()[-1] if r.stderr else "failed")

print(f"\n{'All required checks passed.' if not failures else f'{failures} required check(s) failed.'}")
sys.exit(1 if failures else 0)
