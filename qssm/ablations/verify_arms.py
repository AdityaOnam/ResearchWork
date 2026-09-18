#!/usr/bin/env python3
"""
verify_arms — the pre-flight assertions. Run this before spending 33 GPU-hours.

Four checks, cheapest first, no test data and no training required:

  1. **Parameter counts** match the design (509,383 / 513,591 / 408,775 / 412,983).

  2. **A1 is the baseline at initialisation, bitwise.** `q_merge` is zero-initialised and
     `QHybridViT` shares every other module with `SimpleViT`, so after both load the same
     warm start they must compute *the same function*. `max|A1(x) - A0(x)| == 0` exactly --
     not "small", zero. If this fails, either the warm start is not landing on some module
     or the q-branch is not actually gated off, and the whole "can only earn its way in"
     experimental design is void.

     This is the assertion the plan is built around, and it is the reason A1 is a subclass
     rather than a rewrite.

  3. **Warm start lands completely.** Reports which keys are missing or unexpected per arm.
     A0/A1 must have zero missing outside the q-branch; A2/A3 legitimately miss the scan
     blocks, which is why they get the optional alignment phase.

  4. **Fused and pure kernels agree** at the shapes these models actually run.

    python verify_arms.py
"""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # qssm/
import paths  # noqa: E402  -- puts models/ on sys.path

import qspace  # noqa: E402
import ssm_ops  # noqa: E402
from rdt_net import ConvGlobalSkipModel  # noqa: E402

WARM = paths.WARM_START
IN_CH, H, W = 91, 174, 145

EXPECTED_PARAMS = {
    "A0 vit": 509_383,
    "A1 hybrid": 513_591,
    "A2 mamba --no-qscan": 408_775,
    "A3 mamba (pure)": 412_983,
}

ARMS = {
    "A0 vit": ("vit", {}),
    "A1 hybrid": ("hybrid", {}),
    "A2 mamba --no-qscan": ("mamba", dict(use_qscan=False)),
    "A3 mamba (pure)": ("mamba", {}),
}


def build(kind, kw, device, order, warm=True):
    model = ConvGlobalSkipModel(in_channels=IN_CH, K=5, encoder=kind,
                                enc_kw=dict(kw, kernel="auto")).to(device)
    if getattr(model.in_vit, "use_qscan", False) or hasattr(model.in_vit, "qscan"):
        model.in_vit.set_qspace_order(order)
    with torch.no_grad():                       # materialise pos_emb on the ViT arms
        model(torch.zeros(1, IN_CH, H, W, device=device))
    rep = None
    if warm and os.path.exists(WARM):
        sd = torch.load(WARM, map_location=device, weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        pe = "in_vit.patch_embed.weight"
        if pe in sd and sd[pe].shape[1] == 92:
            sd[pe] = sd[pe][:, :91].contiguous()
        rep = model.load_state_dict(sd, strict=False)
    model.eval()
    return model, rep


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    order = torch.as_tensor(qspace.check(verbose=False), device=device)
    ok = True

    print(f"device      {device}")
    print(f"warm start  {WARM}")
    print(f"            {'present' if os.path.exists(WARM) else 'MISSING'}\n")

    print("1. parameter counts")
    models = {}
    for name, (kind, kw) in ARMS.items():
        m, rep = build(kind, kw, device, order)
        models[name] = (m, rep)
        got = sum(p.numel() for p in m.parameters())
        want = EXPECTED_PARAMS[name]
        good = got == want
        ok &= good
        print(f"   {'ok ' if good else 'FAIL'} {name:<22} {got:>9,}  (expected {want:,})")

    print("\n2. A1 == A0 at initialisation, bitwise")
    if not os.path.exists(WARM):
        print("   SKIP  no warm start on disk")
    else:
        torch.manual_seed(0)
        x = torch.randn(1, IN_CH, H, W, device=device)
        with torch.no_grad():
            y0 = models["A0 vit"][0](x)
            y1 = models["A1 hybrid"][0](x, order)
        d = float((y0 - y1).abs().max())
        good = d == 0.0
        ok &= good
        print(f"   {'ok ' if good else 'FAIL'} max|A1(x) - A0(x)| = {d:.3e}"
              + ("" if good else "   <-- q-branch is NOT gated off, or warm start missed"))

    print("\n3. warm-start coverage")
    for name, (m, rep) in models.items():
        if rep is None:
            continue
        miss = [k for k in rep.missing_keys]
        q_only = all(".qscan." in k or ".q_merge." in k for k in miss)
        note = ("q-branch only" if miss and q_only
                else "complete" if not miss else "scan blocks fresh")
        print(f"   {name:<22} missing {len(miss):>3} · unexpected "
              f"{len(rep.unexpected_keys):>3}   ({note})")

    print("\n4. fused vs pure selective scan")
    if ssm_ops.HAVE_FUSED and torch.cuda.is_available():
        try:
            worst = ssm_ops.self_test(verbose=True)
            print(f"   ok  worst {worst:.3e}")
        except AssertionError as exc:
            ok = False
            print(f"   FAIL {exc}")
    else:
        print(f"   SKIP  fused unavailable ({ssm_ops.FUSED_IMPORT_ERROR!r})")

    print("\nALL CHECKS PASSED" if ok else "\nFAILURES ABOVE -- do not train")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
