"""Path resolution for the free-water-corrected CSD tractography pipeline.

Every location comes from an environment variable, so nothing here is machine-specific:

    MRTRIX_BIN     MRtrix3 bin/ directory      (default: rely on PATH)
    SIGNAL_ROOT    <root>/<sid>/T1w/Diffusion/dwi_b0_1000.nii.gz + bvals/bvecs
    MASK_ROOT      <root>/<sid>/T1w/Diffusion/nodif_brain_mask.nii.gz
    FW_PRED_ROOT   <root>/<sid>/predicted_free_water_volume.nii.gz  (model output)
    TRACT_OUT      output root                  (default: ./tractography_results)
    SUBJECTS_FILE  one subject ID per line      (default: <repo>/splits/test.txt)
"""

import os
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _p(name: str, default: str = "") -> Path:
    return Path(os.path.expanduser(os.environ.get(name, default)))


MRTRIX_BIN = _p("MRTRIX_BIN")
SIGNAL_ROOT = _p("SIGNAL_ROOT")
MASK_ROOT = _p("MASK_ROOT")
A3_PRED_ROOT = _p("FW_PRED_ROOT")
OUT_ROOT = _p("TRACT_OUT", "./tractography_results")


def _subjects() -> list:
    f = _p("SUBJECTS_FILE", str(_REPO / "splits" / "test.txt"))
    if not f.is_file():
        return []
    return [ln.split("#", 1)[0].strip() for ln in f.read_text().splitlines()
            if ln.split("#", 1)[0].strip()]


SUBJECTS = _subjects()


@dataclass(frozen=True)
class SubjectPaths:
    sid: str
    dwi_nii: Path
    bvals: Path
    bvecs: Path
    mask_nii: Path
    fwv_nii: Path
    out_dir: Path

    @classmethod
    def for_subject(cls, sid: str) -> "SubjectPaths":
        sd = SIGNAL_ROOT / sid / "T1w" / "Diffusion"
        md = MASK_ROOT / sid / "T1w" / "Diffusion"
        # Two naming conventions exist for the same data: "dwi_b0_1000.bvals"/".bvecs"
        # (true acquired values) and "bvals_b0_1000"/"bvecs_b0_1000" (pre-rounded to
        # 0/1000). Prefer the former, fall back to the latter.
        bvals = sd / "dwi_b0_1000.bvals"
        bvecs = sd / "dwi_b0_1000.bvecs"
        if not bvals.exists():
            bvals = sd / "bvals_b0_1000"
        if not bvecs.exists():
            bvecs = sd / "bvecs_b0_1000"
        return cls(
            sid=sid,
            dwi_nii=sd / "dwi_b0_1000.nii.gz",
            bvals=bvals,
            bvecs=bvecs,
            mask_nii=md / "nodif_brain_mask.nii.gz",
            fwv_nii=A3_PRED_ROOT / sid / "predicted_free_water_volume.nii.gz",
            out_dir=OUT_ROOT / sid,
        )


def validate(p: SubjectPaths) -> None:
    for f in (p.dwi_nii, p.bvals, p.bvecs, p.mask_nii, p.fwv_nii):
        if not f.exists():
            raise FileNotFoundError(f"[{p.sid}] missing required input: {f}")
