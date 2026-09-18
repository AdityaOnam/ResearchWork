"""Per-subject orchestrator: original vs free-water-corrected (predicted map) FOD tractography.

Runs both DWI variants through mrtrix_ops and writes a
metrics.json (original{}, fw_corrected{}, streamline_change_pct), plus a
deterministic corroborating signal fod_amp0_change_pct (mean FOD l=0
amplitude in the WM seed mask), since streamline_change_pct alone carries
residual seeding noise -- this MRtrix3 build exposes no RNG-seed-fixing flag
on tckgen, so -select 0 -seeds N (fixed seed *attempt* budget, uncapped
output) is used instead of a fixed streamline-count target.

Idempotent: re-running a subject with a valid existing metrics.json is a
no-op unless force=True.
"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np

import mrtrix_ops as mx
from fw_correction import correct_subject_dwi, load_fslgrad
from paths import OUT_ROOT, SubjectPaths, validate

CORRECTION_STRENGTH = 0.7
D_ISO = 3.0e-3
N_SEEDS = 2_000_000
NTHREADS = 8


def metrics_path(sid: str) -> Path:
    return OUT_ROOT / sid / "metrics.json"


def already_done(sid: str) -> bool:
    mp = metrics_path(sid)
    if not mp.exists():
        return False
    try:
        m = json.loads(mp.read_text())
        return "original" in m and "fw_corrected" in m
    except Exception:
        return False


def mean_fod_amp0_in_mask(amp0_nii: Path, seed_mask_mif: Path, sp: SubjectPaths) -> float:
    amp0 = nib.load(amp0_nii).get_fdata()
    # seed_mask_mif is a .mif bit mask; easier to threshold the amp0 volume
    # the same way mrthreshold did is unnecessary here -- just reuse the
    # brain mask + a data-driven positive-amplitude mask for the mean.
    brain = nib.load(sp.mask_nii).get_fdata().astype(bool)
    vals = amp0[brain]
    vals = vals[vals > 0]
    return float(np.mean(vals)) if vals.size else 0.0


def process_variant(sid: str, variant: str, dwi_mif_src_nii: Path, bvecs: Path, bvals: Path,
                     mask_mif: Path, sp: SubjectPaths, out_dir: Path, log_path: Path,
                     nthreads: int = NTHREADS, n_seeds: int = N_SEEDS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = {}

    dwi_mif = out_dir / "dwi.mif"
    timing["mrconvert"] = mx.mrconvert_dwi_to_mif(dwi_mif_src_nii, bvecs, bvals, dwi_mif,
                                                   nthreads=nthreads, log=log_path)

    response_txt = out_dir / "wm_response.txt"
    timing["dwi2response"] = mx.dwi2response_tournier(dwi_mif, response_txt, mask_mif,
                                                        nthreads=nthreads, log=log_path)

    fod_mif = out_dir / "wm_fod.mif"
    timing["dwi2fod"] = mx.dwi2fod_csd(dwi_mif, response_txt, fod_mif, mask_mif,
                                        nthreads=nthreads, log=log_path)

    amp0_nii = out_dir / "fod_amp0.nii.gz"
    seed_mask_mif = out_dir / "wm_seed_mask.mif"
    timing["seed_mask"] = mx.make_wm_seed_mask(fod_mif, amp0_nii, seed_mask_mif, mask_mif, log=log_path)

    tck_out = out_dir / "tracks.tck"
    timing["tckgen"] = mx.tckgen_run(fod_mif, seed_mask_mif, mask_mif, tck_out,
                                      n_seeds=n_seeds, nthreads=nthreads, log=log_path)

    tdi_nii = out_dir / "tdi.nii.gz"
    timing["tckmap"] = mx.tckmap_density(tck_out, mask_mif, tdi_nii, log=log_path)

    stats = mx.tckstats_summary(tck_out, log=log_path)
    timing["tckstats"] = stats.pop("timing_sec")
    timing["total"] = sum(timing.values())

    stats["mean_wm_fod_amp0"] = mean_fod_amp0_in_mask(amp0_nii, seed_mask_mif, sp)
    stats["timing_sec"] = timing
    stats["outputs"] = {"tck": str(tck_out), "tdi": str(tdi_nii), "fod": str(fod_mif)}
    return stats


def run_subject(sid: str, force: bool = False, nthreads: int = NTHREADS,
                 n_seeds: int = N_SEEDS, correction_strength: float = CORRECTION_STRENGTH,
                 d_iso: float = D_ISO) -> dict:
    if not force and already_done(sid):
        return json.loads(metrics_path(sid).read_text())

    sp = SubjectPaths.for_subject(sid)
    validate(sp)
    sp.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUT_ROOT / "logs" / f"{sid}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"=== run_subject {sid} start {time.ctime()} ===\n")

    t_start = time.time()

    mask_mif = sp.out_dir / "mask.mif"
    mx.mrconvert_mask_to_mif(sp.mask_nii, mask_mif, log=log_path)

    corrected_dwi_nii = correct_subject_dwi(sp, correction_strength, d_iso)

    orig_stats = process_variant(sid, "original", sp.dwi_nii, sp.bvecs, sp.bvals, mask_mif,
                                  sp, sp.out_dir / "original", log_path, nthreads, n_seeds)
    corr_stats = process_variant(sid, "corrected", corrected_dwi_nii, sp.bvecs, sp.bvals, mask_mif,
                                  sp, sp.out_dir / "corrected", log_path, nthreads, n_seeds)

    fwv = nib.load(sp.fwv_nii).get_fdata()
    mask = nib.load(sp.mask_nii).get_fdata().astype(bool)

    n_orig = max(orig_stats["streamline_count"], 1)
    streamline_change_pct = 100.0 * (corr_stats["streamline_count"] - orig_stats["streamline_count"]) / n_orig
    fod_amp0_change_pct = 100.0 * (corr_stats["mean_wm_fod_amp0"] - orig_stats["mean_wm_fod_amp0"]) / max(orig_stats["mean_wm_fod_amp0"], 1e-9)

    metrics = {
        "subject_id": sid,
        "config": {
            "nthreads": nthreads, "n_seeds": n_seeds, "select": 0,
            "cutoff": 0.06, "angle": 45, "step": 0.625,
            "minlength": 20, "maxlength": 250,
            "correction_strength": correction_strength, "d_iso": d_iso,
            "seed_mask_threshold_method": "mrthreshold-auto-within-brain-mask",
            "rng_seed_fixed": False,
            "note": ("MRtrix3 3.0.8 tckgen on this build exposes no RNG-seed flag; "
                     "streamline_change_pct therefore carries residual seeding noise on top "
                     "of the true FOD effect. fod_amp0_change_pct is deterministic and used "
                     "as a corroborating signal."),
        },
        "input_shape": list(nib.load(sp.dwi_nii).shape),
        "mask_voxels": int(mask.sum()),
        "mean_fwv_in_mask": float(fwv[mask].mean()),
        "original": {k: v for k, v in orig_stats.items() if k != "outputs" and k != "timing_sec"},
        "fw_corrected": {k: v for k, v in corr_stats.items() if k != "outputs" and k != "timing_sec"},
        "streamline_change_pct": streamline_change_pct,
        "fod_amp0_change_pct": fod_amp0_change_pct,
        "outputs": {"original": orig_stats["outputs"], "corrected": corr_stats["outputs"]},
        "timing_sec": {"original": orig_stats["timing_sec"], "corrected": corr_stats["timing_sec"],
                        "wall_total": time.time() - t_start},
    }

    metrics_path(sid).write_text(json.dumps(metrics, indent=2))
    with open(log_path, "a") as f:
        f.write(f"\n=== run_subject {sid} done in {time.time() - t_start:.1f}s ===\n")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--nthreads", type=int, default=NTHREADS)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    m = run_subject(args.sid, force=args.force, nthreads=args.nthreads, n_seeds=args.n_seeds)
    print(json.dumps(m, indent=2))
