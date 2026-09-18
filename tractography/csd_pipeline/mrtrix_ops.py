"""MRtrix3 subprocess wrappers for FOD estimation and tractography.

Follows the CLI-wrapping pattern already used in
MCOVM_Framework/Stage1_Tissue_Decomposition/tractography_hcp.py, and reuses
its tracking parameters (-minlength 20 -maxlength 250 -cutoff 0.06 -angle 45
-step 0.625) for consistency with the project's existing tractography
outputs.

Since the input DWI is genuinely single-shell (b=1000 + b0), FOD estimation
uses single-tissue CSD (dwi2response tournier -> dwi2fod csd), not the
multi-shell dhollander/msmt_csd recipe.
"""

import subprocess
import time
from pathlib import Path

from paths import MRTRIX_BIN


def run(cmd: list, log_path: Path | None = None) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    cmd[0] = str(MRTRIX_BIN / cmd[0])
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if log_path:
        with open(log_path, "a") as f:
            f.write(f"\n$ {' '.join(cmd)}\n[{dt:.1f}s]\n{result.stdout}\n{result.stderr}\n")
    if result.returncode != 0:
        raise RuntimeError(f"MRtrix command failed ({dt:.1f}s): {' '.join(cmd)}\n{result.stderr}")
    return result, dt


def mrconvert_dwi_to_mif(nii: Path, bvecs: Path, bvals: Path, mif_out: Path,
                          nthreads: int = 8, log: Path | None = None) -> float:
    _, dt = run(["mrconvert", nii, mif_out, "-fslgrad", bvecs, bvals,
                 "-datatype", "float32", "-nthreads", nthreads, "-force"], log)
    return dt


def mrconvert_mask_to_mif(nii: Path, mif_out: Path, log: Path | None = None) -> float:
    _, dt = run(["mrconvert", nii, mif_out, "-datatype", "bit", "-force"], log)
    return dt


def dwi2response_tournier(dwi_mif: Path, response_txt: Path, mask_mif: Path,
                           nthreads: int = 8, log: Path | None = None) -> float:
    _, dt = run(["dwi2response", "tournier", dwi_mif, response_txt,
                 "-mask", mask_mif, "-nthreads", nthreads, "-force"], log)
    return dt


def dwi2fod_csd(dwi_mif: Path, response_txt: Path, fod_mif: Path, mask_mif: Path,
                 nthreads: int = 8, log: Path | None = None) -> float:
    _, dt = run(["dwi2fod", "csd", dwi_mif, response_txt, fod_mif,
                 "-mask", mask_mif, "-nthreads", nthreads, "-force"], log)
    return dt


def make_wm_seed_mask(fod_mif: Path, amp0_out: Path, seed_mask_out: Path,
                       mask_mif: Path, log: Path | None = None) -> float:
    _, dt1 = run(["mrconvert", fod_mif, "-coord", "3", "0", amp0_out, "-force"], log)
    _, dt2 = run(["mrthreshold", amp0_out, seed_mask_out, "-mask", mask_mif, "-force"], log)
    return dt1 + dt2


def tckgen_run(fod_mif: Path, seed_mask_mif: Path, mask_mif: Path, tck_out: Path,
                n_seeds: int = 2_000_000, nthreads: int = 8, log: Path | None = None) -> float:
    _, dt = run(["tckgen", fod_mif, tck_out,
                 "-algorithm", "iFOD2",
                 "-seed_image", seed_mask_mif,
                 "-mask", mask_mif,
                 "-select", 0, "-seeds", n_seeds,
                 "-minlength", 20, "-maxlength", 250,
                 "-cutoff", 0.06, "-angle", 45, "-step", 0.625,
                 "-nthreads", nthreads, "-force"], log)
    return dt


def tckmap_density(tck_in: Path, template_mif: Path, tdi_out_nii: Path,
                    log: Path | None = None) -> float:
    _, dt = run(["tckmap", tck_in, "-template", template_mif, tdi_out_nii, "-force"], log)
    return dt


def tckstats_summary(tck_in: Path, log: Path | None = None) -> dict:
    result, dt = run(["tckstats", tck_in, "-output", "count",
                       "-output", "mean", "-output", "median"], log)
    parts = result.stdout.split()
    count, mean_len, median_len = (float(x) for x in parts[:3])
    return {
        "streamline_count": int(count),
        "mean_length_mm": mean_len,
        "median_length_mm": median_len,
        "timing_sec": dt,
    }
