#!/usr/bin/env python3
"""
evaluate_snr — robustness sweep: every model against SNR 60 -> 10, plus clean.

Reuses `evaluate.py` for model loading, preprocessing, masking and scoring, so an SNR row
is produced by exactly the same code path as the clean number and the two are directly
comparable. The clean column IS the SNR=infinity point of the same curve.

Noise comes from `noise_snr.py` — the Rician formula shared with hpc/adaptive_compute, applied slice-wise to the
raw intensity before b0 normalisation, deterministically seeded per (subject, SNR). See
that module for why the pre-generated `noisy_snr_test/` set could not be used here
(multi-shell input and AMICO ground truth, neither of which matches these models).

Metric is `torch.mean(|pred*m - gt*m|)` over all 174x145 pixels — the only convention this
project reports. Ground truth is the DIPY bi-tensor map, the same one the clean numbers
use, so degradation is measured against a fixed reference.

Each subject's raw DWI volume is loaded from disk **once**, not once per SNR level: an
earlier version called this inside the (subject, snr) loop and profiling
(evaluate_directions.py, the same pattern) showed most of the wall time was disk IO, not
GPU compute. Noise is then drawn six times (once per level) from the cached raw array in
memory. The clean column still goes through `evaluate.score_subject` verbatim, unchanged
from the original script, so it stays provably identical to the canonical clean numbers.

    python evaluate_snr.py --all
    python evaluate_snr.py --model A3_pure --snr 30 20 10 --subjects 6
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

import nibabel as nib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # qssm/
import paths  # noqa: E402  -- puts models/, data/, evaluate/ on sys.path
import evaluate as ev  # noqa: E402
from noise_snr import SNR_LEVELS, noisify_volume  # noqa: E402


# label -> (checkpoint, dt_live). dt_live cannot be inferred from weights; see evaluate.py.
MODELS = {
    "A0_vit":        (paths.ckpt(paths.ARM_CKPTS["A0"]), False),
    "A1_hybrid":     (paths.ckpt(paths.ARM_CKPTS["A1"]), False),
    "A2_mamba_noqs": (paths.ckpt(paths.ARM_CKPTS["A2"]), False),
    "A3_pure":       (paths.ckpt(paths.ARM_CKPTS["A3"]), False),
    "A1_dtlive":     (paths.ckpt("VITQM_pde_K5_Rxn_diff_synth_hybrid_r70_guided_dtlive.pth"), True),
    "baseline_92ch": (paths.ckpt("VIT_pde_K5_Rxn_diff.pth"), False),
}
OUT = os.path.join(ev.OUT_DIR, "snr_sweep.csv")


@torch.no_grad()
def score_noisy_levels(model, in_ch, sid, snr_levels):
    """One subject, ALL noisy SNR levels, one disk read. Returns {snr: mae_or_None}."""
    dwi = f"{ev.SIGNAL}/{sid}/T1w/Diffusion/dwi_b0_1000.nii.gz"
    msk = f"{ev.MASK}/{sid}/T1w/Diffusion/nodif_brain_mask.nii.gz"
    gtp = f"{ev.GT}/{sid}/free_water_volume.nii.gz"
    pri = f"{ev.PRIOR}/{sid}/dwi_b0_1000/fw_volume_fraction.nii.gz"
    need = [dwi, msk, gtp] + ([pri] if in_ch == 92 else [])
    if any(not os.path.exists(p) for p in need):
        return {snr: None for snr in snr_levels}

    img = nib.load(dwi)
    raw = np.asarray(img.dataobj, dtype=np.float32)
    mask, gt = ev.load3d(msk), ev.load3d(gtp)
    fc = ev.load3d(pri) if in_ch == 92 else None

    out = {}
    for snr in snr_levels:
        noisy = noisify_volume(raw, snr, sid)

        # b0-normalise exactly as prepare_dwi_b0_flat_and_slices does: divide by volume 0,
        # clip to [0,1]. Done here because the noise had to go in before this step.
        b0 = noisy[..., 0:1].astype(np.float64)
        b0[np.abs(b0) < 1e-10] = 1e-10
        sig = np.clip(noisy.astype(np.float64) / b0, 0.0, 1.0).astype(np.float32)

        script = []
        for i in range(sig.shape[2]):                       # axial slices, Z axis
            d = sig[:, :, i, :].transpose(2, 0, 1)          # (V, H, W)
            x = d if in_ch == 91 else np.concatenate([d, fc[:, :, i][None]], 0)
            m = (mask[:, :, i] > 0.5).astype(np.float32)
            x = x * m[None]                                 # --mask-input, always on
            p = torch.clamp(model(torch.from_numpy(x[None]).to(ev.device)), 0, 1)
            p = p.squeeze().cpu().numpy()
            script.append(float(np.abs(p * m - gt[:, :, i] * m).mean()))
        out[snr] = float(np.mean(script))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[],
                    help=f"one of {sorted(MODELS)}; repeatable")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--snr", type=float, nargs="*", default=list(SNR_LEVELS))
    ap.add_argument("--subjects", type=int, default=0, help="limit (quick look)")
    args = ap.parse_args()

    names = sorted(MODELS) if args.all else args.model
    if not names:
        ap.error("pass --model NAME (repeatable) or --all")
    subs = list(ev.test_subjects())[:args.subjects or None]
    noisy_levels = sorted(args.snr, reverse=True)

    print(f"SNR sweep · {len(subs)} subjects · torch.mean MAE · DIPY bi-tensor GT")
    print("noise: Rician, slice-wise, seeded per (subject, SNR); clean = same path as "
          "evaluate.py; one disk read per subject, all SNR levels reselected in memory\n")

    rows = []
    for name in names:
        path, live = MODELS[name]
        if not os.path.exists(path):
            print(f"  {name:<15} checkpoint missing — skipped")
            continue
        model, in_ch, _ = ev.load_model(path, dt_live=live)

        # Clean: unchanged path, one call per subject, provably the canonical number.
        clean_vals = []
        for sid in subs:
            res, err = ev.score_subject(model, in_ch, sid, None, True)
            if not err:
                clean_vals.append(res[1])

        per_level_vals = {snr: [] for snr in noisy_levels}
        for sid in subs:
            res = score_noisy_levels(model, in_ch, sid, noisy_levels)
            for snr, mae in res.items():
                if mae is not None:
                    per_level_vals[snr].append(mae)

        line = {"model": name, "in_channels": in_ch}
        line["clean"] = float(np.mean(clean_vals)) if clean_vals else float("nan")
        cells = [f"{line['clean']:.6f}"]
        for snr in noisy_levels:
            vals = per_level_vals[snr]
            mae = float(np.mean(vals)) if vals else float("nan")
            line[f"snr{int(snr)}"] = mae
            cells.append(f"{mae:.6f}")
        rows.append(line)

        hdr = "  ".join(f"{s:>8}" for s in (["clean"] + [int(x) for x in noisy_levels]))
        if len(rows) == 1:
            print(f"  {'model':<15}{hdr}")
        print(f"  {name:<15}" + "  ".join(f"{c:>8}" for c in cells))
        del model
        torch.cuda.empty_cache()

    if rows:
        os.makedirs(ev.OUT_DIR, exist_ok=True)
        cols = ["model", "in_channels", "clean"] + [f"snr{int(s)}" for s in noisy_levels]
        with open(OUT, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {OUT}   ({datetime.now():%Y-%m-%d %H:%M})")


if __name__ == "__main__":
    main()
