#!/usr/bin/env python3
"""
evaluate — score a checkpoint on the held-out test subjects against the DIPY bi-tensor
ground truth, reporting both MAE conventions.

Handles 91- and 92-channel checkpoints through one path, so the QSSM arms and the
RDT-Net baseline are measured by identical code on identical inputs. The channel count
is read off the checkpoint's patch embedding; a 92-channel model additionally loads the
Stage-I ANN prior for channel 92.

**The metric is `torch.mean(|pred*m - gt*m|)` and nothing else.** The mask premultiply
zeroes the background in the numerator, but the divisor stays the full 174x145 frame. It is
what the trainer optimises, what checkpoint selection uses, and the only number that goes in
a table or a paper. Reported here as `MAE`.

An in-mask figure (`sum|pred - gt| over brain / brain voxel count`) is still *computed* and
still written to `summary.csv`: being brain-fraction-independent it is a useful
diagnostic when two subjects disagree. It is deliberately absent from the printed banner and
plays no part in any ranking. Do not put it in a table.

`--mask-input` **defaults on**, and is read from the checkpoint's own metadata when present.
The corpus zeroes the input outside the brain at load, so a model fed raw signal at test
time sees a different distribution than it trained on, and scores noticeably worse.

The encoder is inferred from the checkpoint's keys, so vit / hybrid / mamba checkpoints all
score through one path on identical inputs.

    python evaluate.py --all
    python evaluate.py --checkpoint $QSSM_CKPT_DIR/<checkpoint>.pth --label A3_pure
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
import paths  # noqa: E402  -- also puts models/ on sys.path
from rdt_net import ConvGlobalSkipModel, prepare_dwi_b0_flat_and_slices  # noqa: E402

# PAFNet / VIT-ADV-DIFF is a different ARCHITECTURE, not different weights: its PDE block
# is advection-diffusion with a 16-channel conv_dt bottleneck where reaction-diffusion has
# 64. Its class is loaded lazily from its own tree so both families can be scored by this
# one evaluator on identical inputs, preprocessing and ground truth.
# Optional: set QSSM_ADV_DIFF_SRC to a PAFNet source tree to score advection-diffusion checkpoints.
ADV_DIFF_SRC = os.environ.get("QSSM_ADV_DIFF_SRC", "")


def _load_adv_diff_class():
    """Import PAFNet's ConvGlobalSkipModel without letting its module shadow ours."""
    import importlib.util
    if not ADV_DIFF_SRC:
        raise RuntimeError("advection-diffusion checkpoint: set QSSM_ADV_DIFF_SRC")
    saved = sys.modules.copy()
    sys.path.insert(0, ADV_DIFF_SRC)
    try:
        spec = importlib.util.spec_from_file_location(
            "_pafnet_test", os.path.join(ADV_DIFF_SRC, "test_vit_pde_enhanced.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.ConvGlobalSkipModel
    finally:
        sys.path.remove(ADV_DIFF_SRC)
        for k in list(sys.modules):
            if k not in saved:
                del sys.modules[k]
        sys.modules.update(saved)

# Test-cohort layout (see docs/data_layout.md).
NEW = paths.TEST_ROOT
SIGNAL = f"{NEW}/Signal"
MASK = f"{NEW}/t_data"
GT = f"{NEW}/Analysis/multi_shell/Ground_Truth"
PRIOR = paths.PRIOR_ROOT


def test_subjects():
    return list(paths.subjects("test"))

K = 5
N_VOLS = 91
OUT_DIR = paths.results("eval")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load3d(p):
    d = np.asarray(nib.load(p).dataobj, np.float32)
    return np.squeeze(d, -1) if d.ndim == 4 and d.shape[-1] == 1 else d


def infer_encoder(sd):
    """Which encoder produced this checkpoint, from its keys alone.

    `scan_blocks.*` only exists on the pure-Mamba encoder; `qscan.*` only exists when a
    q-space branch is present. So the three arms are distinguishable without metadata,
    which matters for checkpoints written before metadata was recorded.
    """
    has_scan = any(k.startswith("in_vit.scan_blocks.") for k in sd)
    has_q = any(k.startswith("in_vit.qscan.") for k in sd)
    if has_scan:
        return "mamba", has_q
    return ("hybrid" if has_q else "vit"), has_q


def load_model(path, kernel="auto", dt_live=None):
    """Returns (model, in_channels, meta). Architecture is read off the checkpoint.

    `dt_live` CANNOT be inferred from weights: Softplus is parameter-free, so a dt-live
    checkpoint and a dead-dt one have byte-identical state dicts. Loading one into the
    other therefore succeeds with `missing 0 / unexpected 0` and silently computes the
    wrong function. Metadata is used when present; otherwise pass --dt-live.
    """
    raw = torch.load(path, map_location=device, weights_only=False)
    meta = raw if isinstance(raw, dict) and "model" in raw else {}
    sd = raw.get("model", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    pe = "in_vit.patch_embed.weight"
    if pe not in sd:
        raise KeyError(f"{path}: no {pe}; cannot infer input channels")
    in_ch = sd[pe].shape[1]

    encoder, has_q = infer_encoder(sd)
    live = meta.get("dt_live", dt_live) if dt_live is None else dt_live
    live = bool(live)
    enc_kw = dict(kernel=kernel)
    if encoder == "mamba":
        enc_kw["use_qscan"] = has_q
    model = ConvGlobalSkipModel(in_channels=in_ch, K=K,
                                encoder=encoder, enc_kw=enc_kw, dt_live=live).to(device)

    if has_q:
        # The traversal is not in the state dict -- it is a property of the acquisition
        # scheme, not of the weights -- so it must be reinstalled here or the scan runs in
        # file order, which is worse than random.
        import qspace
        model.in_vit.set_qspace_order(qspace.check(verbose=False))

    # pos_emb is built inside forward(); materialise it first or strict=False reports it
    # "unexpected" and silently drops 99k trained parameters. No-op for the mamba arms.
    with torch.no_grad():
        model(torch.zeros(1, in_ch, 174, 145, device=device))
    r = model.load_state_dict(sd, strict=False)
    if r.unexpected_keys or r.missing_keys:
        print(f"    [warn] missing {r.missing_keys} · unexpected {r.unexpected_keys}")
    model.eval()
    meta["encoder"] = encoder
    meta["qscan"] = has_q
    meta["dt_live"] = live
    return model, in_ch, meta


@torch.no_grad()
def score_subject(model, in_ch, sid, save_dir=None, mask_input=False):
    dwi = f"{SIGNAL}/{sid}/T1w/Diffusion/dwi_b0_1000.nii.gz"
    msk = f"{MASK}/{sid}/T1w/Diffusion/nodif_brain_mask.nii.gz"
    gtp = f"{GT}/{sid}/free_water_volume.nii.gz"
    pri = f"{PRIOR}/{sid}/dwi_b0_1000/fw_volume_fraction.nii.gz"

    need = [dwi, msk, gtp] + ([pri] if in_ch == 92 else [])
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        return None, f"missing {os.path.basename(missing[0])}"

    sig, dims, affine, header, _, _, _ = prepare_dwi_b0_flat_and_slices(dwi, N_VOLS, None)
    S, C, H, W = sig.shape
    mask, gt = load3d(msk), load3d(gtp)
    fc = load3d(pri) if in_ch == 92 else None

    preds, script = [], []
    for i in range(S):
        d = sig[i][:N_VOLS] if C >= N_VOLS else np.concatenate(
            [sig[i], np.zeros((N_VOLS - C, H, W), np.float32)], 0)
        x = d if in_ch == 91 else np.concatenate([d, fc[i][None]], 0)
        if mask_input:
            # Must match how the checkpoint was trained. A model trained on
            # brain-only input sees a different distribution if fed raw signal at
            # test time, and vice versa -- this flag is not cosmetic.
            x = x * (mask[i] > 0.5).astype(np.float32)[None]
        p = torch.clamp(model(torch.from_numpy(x[None].astype(np.float32)).to(device)), 0, 1)
        p = p.squeeze().cpu().numpy()
        preds.append(p)
        m = (mask[i] > 0.5).astype(np.float32)
        script.append(float(np.abs(p * m - gt[i] * m).mean()))

    pred = np.stack(preds, 0).astype(np.float32)
    brain = mask > 0.5
    in_mask = float(np.abs(pred[brain] - gt[brain]).mean())

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        ref = nib.load(dwi)
        nib.save(nib.Nifti1Image((pred * brain).astype(np.float32), ref.affine),
                 os.path.join(save_dir, "predicted_free_water_volume.nii.gz"))
        nib.save(nib.Nifti1Image(((1 - pred) * brain).astype(np.float32), ref.affine),
                 os.path.join(save_dir, "tissue_volume_fraction.nii.gz"))

    return (in_mask, float(np.mean(script)), int(brain.sum())), None


def evaluate(path, label, save_maps=False, mask_input=True, kernel="auto", dt_live=None):
    print(f"\n=== {label} ===\n  {path}")
    model, in_ch, meta = load_model(path, kernel, dt_live)
    # A checkpoint that recorded how it was trained overrides the flag -- the flag is a
    # default for older files, not a licence to score a model on the wrong distribution.
    if "mask_input" in meta:
        if meta["mask_input"] != mask_input:
            print(f"  mask-input {mask_input} -> {meta['mask_input']} (from checkpoint)")
        mask_input = meta["mask_input"]
    enc = meta.get("encoder", "?")
    print(f"  encoder {enc}{' +qscan' if meta.get('qscan') else ''} · {in_ch} channels · "
          f"{sum(p.numel() for p in model.parameters()):,} params · "
          f"mask-input {mask_input} · dt-live {meta.get('dt_live')}"
          + ("  (uses the ANN prior at ch 92)" if in_ch == 92 else ""))

    rows = []
    for sid in test_subjects():
        sd = os.path.join(OUT_DIR, label, sid) if save_maps else None
        res, err = score_subject(model, in_ch, sid, sd, mask_input)
        if err:
            print(f"  {sid}  SKIP ({err})")
            continue
        im, sc, n = res
        rows.append((sid, im, sc, n))
        print(f"  {sid}  MAE {sc:.6f}   (n={n:,})")

    a = np.array([r[1] for r in rows]); b = np.array([r[2] for r in rows])
    # in-mask (`a`) is written to CSV but never printed and never ranked on -- see the
    # module docstring.
    print(f"  {'-'*58}\n  N={len(rows)}   MAE {b.mean():.6f} ± {b.std(ddof=1):.6f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{label}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "mae_inmask", "mae_script", "brain_voxels"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["mean", f"{a.mean():.6f}", f"{b.mean():.6f}", ""])
        w.writerow(["std", f"{a.std(ddof=1):.6f}", f"{b.std(ddof=1):.6f}", ""])
    return dict(label=label, checkpoint=path, in_channels=in_ch, n=len(rows),
                inmask_mean=float(a.mean()), inmask_std=float(a.std(ddof=1)),
                script_mean=float(b.mean()), script_std=float(b.std(ddof=1)))


SUMMARY_FIELDS = ["label", "checkpoint", "in_channels", "n",
                  "inmask_mean", "inmask_std", "script_mean", "script_std", "when"]


def merge_summary(path, rows):
    """Upsert one row per label, preserving every label this run did not touch.

    Upsert rather than append, or re-scoring a label would accumulate duplicates.
    Written via a temp file + os.replace so an interrupted run cannot truncate it.
    """
    by_label = {}
    if os.path.exists(path):
        with open(path, newline="") as fh:
            by_label = {r["label"]: r for r in csv.DictReader(fh)}
    stamp = datetime.now().isoformat(timespec="seconds")
    for r in rows:
        by_label[r["label"]] = {**r, "when": stamp}
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(by_label.values())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", default=[])
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--all", action="store_true",
                    help="baseline + both synth checkpoints, whichever exist")
    ap.add_argument("--save-maps", action="store_true")
    # Defaults ON: the corpus masks the input at load, so an unmasked eval scores the model
    # on a distribution it never trained on. A checkpoint carrying
    # its own `mask_input` overrides this either way.
    ap.add_argument("--mask-input", dest="mask_input", action="store_true", default=True,
                    help="zero the input outside the brain mask (default)")
    ap.add_argument("--no-mask-input", dest="mask_input", action="store_false",
                    help="feed raw signal; only for checkpoints trained unmasked")
    ap.add_argument("--ssm-kernel", default="auto", dest="ssm_kernel",
                    choices=("auto", "fused", "torch"))
    ap.add_argument("--dt-live", dest="dt_live", action="store_true", default=None,
                    help="checkpoint was trained with --dt-live. Cannot be inferred from "
                         "weights; getting it wrong silently scores the wrong function.")
    args = ap.parse_args()

    jobs = list(zip(args.checkpoint, args.label + [None] * len(args.checkpoint)))
    if args.all:
        ck = paths.ckpt
        cand = [
            # reference points, scored by this same code on the same inputs
            (ck("VIT_pde_K5_Rxn_diff.pth"), "RDT_NET_baseline"),
            (paths.WARM_START, "synth_r70_guided"),
            # the 2x2 factorial
            (ck(paths.ARM_CKPTS["A0"]), "A0_vit"),
            (ck(paths.ARM_CKPTS["A1"]), "A1_hybrid"),
            (ck(paths.ARM_CKPTS["A2"]), "A2_mamba_noqs"),
            (ck(paths.ARM_CKPTS["A3"]), "A3_pure"),
        ]
        jobs += [(p, l) for p, l in cand if os.path.exists(p)]
    if not jobs:
        ap.error("nothing to evaluate: pass --checkpoint or --all")

    summary = []
    for path, label in jobs:
        label = label or os.path.splitext(os.path.basename(path))[0]
        summary.append(evaluate(path, label, args.save_maps, args.mask_input,
                                args.ssm_kernel, args.dt_live))

    print("\n" + "=" * 78)
    print(f"{'model':34} {'ch':>3} {'N':>3} {'MAE':>22}")
    for s in sorted(summary, key=lambda r: r["script_mean"]):
        print(f"{s['label']:34} {s['in_channels']:>3} {s['n']:>3} "
              f"{s['script_mean']:>13.6f} ± {s['script_std']:.6f}")
    print("=" * 78)
    print(f"{len(test_subjects())} test subjects · b=1000 · GT = DIPY multi-shell bi-tensor · "
          f"{datetime.now():%Y-%m-%d %H:%M}")

    merge_summary(os.path.join(OUT_DIR, "summary.csv"), summary)
    print(f"\nwritten to {OUT_DIR}/")


if __name__ == "__main__":
    main()
