#!/usr/bin/env python3
"""
dwi_preprocess — raw BIDS DWI (AP/PA pair + T1w) to a model-ready, T1-registered DWI.

Pipeline (MRtrix3 + FSL + ANTs):

    mrconvert (import BIDS gradients + JSON)
    dwidenoise -> mrdegibbs                         MP-PCA denoising, Gibbs ringing removal
    dwifslpreproc -rpe_pair                         topup + eddy (with eddy QC)
    dwibiascorrect ants                             N4 bias-field correction
    dwi2mask                                        brain mask
    mrconvert -export_grad_fsl                      dwi.nii.gz + bvals/bvecs
    mrregister -type rigid + mrtransform            T1w -> DWI space

Outputs (in --out-dir): dwi.nii.gz, bvals, bvecs, dwi_brain_mask.nii.gz,
T1_in_DWI_space.nii.gz, eddy_qc/.

    python dwi_preprocess.py --ap sub-X_acq-AP_dwi.nii.gz --ap-grad sub-X_acq-AP_dwi.txt \\
        --ap-json sub-X_acq-AP_dwi.json --pa sub-X_acq-PA_dwi.nii.gz \\
        --pa-json sub-X_acq-PA_dwi.json --t1 sub-X_T1w.nii.gz --out-dir out/sub-X
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(cmd: str, dry: bool):
    print(f"\n>>> {cmd}", flush=True)
    if not dry:
        subprocess.run(cmd, shell=True, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ap", required=True, help="AP-encoded DWI (.nii.gz)")
    ap.add_argument("--ap-grad", required=True, help="MRtrix gradient table for the AP DWI")
    ap.add_argument("--ap-json", required=True, help="BIDS sidecar for the AP DWI")
    ap.add_argument("--pa", required=True, help="PA-encoded b=0 series (.nii.gz)")
    ap.add_argument("--pa-json", required=True, help="BIDS sidecar for the PA series")
    ap.add_argument("--t1", required=True, help="T1-weighted image (.nii.gz)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nthreads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args()

    for f in (a.ap, a.ap_grad, a.ap_json, a.pa, a.pa_json, a.t1):
        if not os.path.exists(f):
            sys.exit(f"missing input: {f}")
    o = a.out_dir
    os.makedirs(o, exist_ok=True)
    nt = f"-nthreads {a.nthreads}"
    R = lambda c: run(c, a.dry_run)  # noqa: E731

    R(f"mrconvert {a.ap} {o}/AP.mif -grad {a.ap_grad} -json_import {a.ap_json} -force")
    R(f"mrconvert {a.pa} {o}/PA.mif -json_import {a.pa_json} -force")
    R(f"mrconvert {a.t1} {o}/T1.mif -force")
    R(f"mrinfo {o}/AP.mif -dwgrad")

    R(f"dwidenoise {o}/AP.mif - {nt} | mrdegibbs - {o}/AP_denoise_unring.mif {nt} -force")
    R(f"dwifslpreproc {o}/AP_denoise_unring.mif {o}/DWI_preproc.mif -rpe_pair "
      f"-se_epi {o}/PA.mif -pe_dir AP -eddyqc_all {o}/eddy_qc {nt} -force")
    R(f"dwibiascorrect ants {o}/DWI_preproc.mif {o}/DWI_preproc_unbiased.mif {nt} -force")
    R(f"dwi2mask {o}/DWI_preproc_unbiased.mif {o}/DWI_mask.mif {nt} -force")

    R(f"mrconvert {o}/DWI_preproc_unbiased.mif {o}/dwi.nii.gz "
      f"-export_grad_fsl {o}/bvecs {o}/bvals -force")
    R(f"mrconvert {o}/DWI_mask.mif {o}/dwi_brain_mask.nii.gz -force")

    R(f"dwiextract {o}/DWI_preproc_unbiased.mif - -bzero | mrmath - mean {o}/mean_b0.mif "
      f"-axis 3 -force")
    R(f"mrregister {o}/T1.mif {o}/mean_b0.mif -type rigid "
      f"-rigid {o}/T1_to_DWI.txt -force")
    R(f"mrtransform {o}/T1.mif -linear {o}/T1_to_DWI.txt -template {o}/mean_b0.mif "
      f"{o}/T1_in_DWI_space.mif -force")
    R(f"mrconvert {o}/T1_in_DWI_space.mif {o}/T1_in_DWI_space.nii.gz -force")
    print("\nDWI preprocessing complete:", o)


if __name__ == "__main__":
    main()
