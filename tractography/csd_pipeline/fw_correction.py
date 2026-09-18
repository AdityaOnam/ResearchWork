"""Free-water DWI signal correction driven by A3's predicted free-water map.

apply_free_water_correction() uses the same formula and defaults as the earlier
PAFNet tractography pipeline. It is DWI-model-agnostic: it only needs dwi_data,
fwv_data, bvals and two scalars. Gradients are read with a plain FSL-format loader,
so dipy is not required.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

from paths import SubjectPaths


def load_fslgrad(bvals_path: Path, bvecs_path: Path) -> tuple[np.ndarray, np.ndarray]:
    bvals = np.loadtxt(bvals_path).reshape(-1)
    bvecs = np.loadtxt(bvecs_path)  # shape (3, N), FSL convention
    return bvals, bvecs


def apply_free_water_correction(dwi_data, fwv_data, bvals, correction_strength, d_iso):
    corrected = np.zeros_like(dwi_data, dtype=np.float32)
    b0_idx = bvals == 0
    dwi_idx = bvals > 0
    if not np.any(b0_idx):
        raise ValueError("At least one b0 volume is required.")
    if not np.any(dwi_idx):
        raise ValueError("At least one diffusion-weighted volume is required.")
    s0 = np.mean(dwi_data[..., b0_idx], axis=-1)
    s0[s0 <= 0] = 1e-6
    corrected[..., b0_idx] = dwi_data[..., b0_idx]
    for vol_idx in np.where(dwi_idx)[0]:
        bval = bvals[vol_idx]
        fw_factor = np.exp(-bval * d_iso)
        signal = dwi_data[..., vol_idx]
        fw_signal = s0 * fwv_data * fw_factor
        signal_corr = np.maximum(signal - (correction_strength * fw_signal), 0)
        tissue_boost = np.clip(1.0 + (0.15 * fwv_data), 1.0, 1.15)
        corrected[..., vol_idx] = signal_corr * tissue_boost
    return corrected


def correct_subject_dwi(sp: SubjectPaths, correction_strength: float = 0.7,
                         d_iso: float = 3.0e-3) -> Path:
    dwi_img = nib.load(sp.dwi_nii)
    dwi_data = dwi_img.get_fdata(dtype=np.float32)
    fwv_data = nib.load(sp.fwv_nii).get_fdata(dtype=np.float32)
    bvals, _ = load_fslgrad(sp.bvals, sp.bvecs)

    if fwv_data.shape != dwi_data.shape[:3]:
        raise ValueError(f"FWV shape {fwv_data.shape} != DWI shape {dwi_data.shape[:3]}")
    if bvals.shape[0] != dwi_data.shape[-1]:
        raise ValueError(f"bvals count {bvals.shape[0]} != DWI volumes {dwi_data.shape[-1]}")

    corrected = apply_free_water_correction(dwi_data, fwv_data, bvals, correction_strength, d_iso)

    out_path = sp.out_dir / "corrected" / "dwi_fw_corrected.nii.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(corrected.astype(np.float32), dwi_img.affine, dwi_img.header), out_path)
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("usage: python fw_correction.py <subject_id>")
    sid = sys.argv[1]
    sp = SubjectPaths.for_subject(sid)
    from paths import validate

    validate(sp)
    out_path = correct_subject_dwi(sp)
    dwi_data = nib.load(sp.dwi_nii).get_fdata(dtype=np.float32)
    corr_data = nib.load(out_path).get_fdata(dtype=np.float32)
    mask = nib.load(sp.mask_nii).get_fdata().astype(bool)
    bvals, _ = load_fslgrad(sp.bvals, sp.bvecs)
    dwi_idx = bvals > 0
    print(f"[{sid}] wrote {out_path}")
    print(f"  shape: {corr_data.shape}, dtype: {corr_data.dtype}")
    print(f"  min: {corr_data.min():.4f}, max: {corr_data.max():.4f}, any negative: {(corr_data < 0).any()}")
    print(f"  mean signal in mask (dwi vols) original:  {dwi_data[..., dwi_idx][mask].mean():.4f}")
    print(f"  mean signal in mask (dwi vols) corrected: {corr_data[..., dwi_idx][mask].mean():.4f}")
