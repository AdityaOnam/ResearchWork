#!/usr/bin/env python3
"""
shell_extractor — split multi-shell HCP DWI into single-shell `b0 + shell` volumes.

For every subject under --base-dir (HCP layout: <sid>/T1w/Diffusion/{data.nii.gz,bvals,bvecs})
this writes, next to the input:

    dwi_b0_<b>.nii.gz, bvals_b0_<b>, bvecs_b0_<b>      for b in --shells (default 1000 2000 3000)

each holding exactly one b=0 volume at index 0 followed by that shell's directions. The
QSSM models consume `dwi_b0_1000.nii.gz` (1 b0 + 90 directions = 91 channels).

Requires MRtrix3 (mrconvert, dwiextract, mrmath, mrcat) on PATH.

    python shell_extractor.py --base-dir /path/to/HCP                  # all subjects
    python shell_extractor.py --base-dir /path/to/HCP --subject SUBJ01  # one (SLURM array)
    python shell_extractor.py --base-dir /path/to/HCP --dry-run         # print commands only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(cmd: str, dry: bool) -> bool:
    print(f"  $ {cmd}", flush=True)
    if dry:
        return True
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [error] {e}", file=sys.stderr)
        return False


def process_subject(base_dir: str, sid: str, shells: list[int], dry: bool, force: bool) -> bool:
    d = os.path.join(base_dir, sid, "T1w", "Diffusion")
    nii, bvecs, bvals = (os.path.join(d, f) for f in ("data.nii.gz", "bvecs", "bvals"))
    if not all(os.path.exists(p) for p in (nii, bvecs, bvals)):
        print(f"[skip] {sid}: data.nii.gz / bvals / bvecs missing")
        return False

    print(f"[{sid}]")
    mif = os.path.join(d, "out.mif")
    b0 = os.path.join(d, "b0_single.mif")
    ok = run(f'mrconvert "{nii}" "{mif}" -fslgrad "{bvecs}" "{bvals}" -force', dry)
    ok &= run(f'dwiextract "{mif}" - -bzero | mrmath - mean "{os.path.join(d, "mean_bzero.mif")}" '
              f'-axis 3 -force', dry)
    # exactly one b=0 (the first), so every shell file starts with the same reference
    ok &= run(f'dwiextract -bzero "{mif}" - | mrconvert -coord 3 0 - "{b0}" -force', dry)

    for b in shells:
        out_nii = os.path.join(d, f"dwi_b0_{b}.nii.gz")
        if os.path.exists(out_nii) and not force:
            print(f"  [done] b={b}")
            continue
        shell = os.path.join(d, f"dwi_shell_{b}.mif")
        cat = os.path.join(d, f"dwi_b0_{b}.mif")
        ok &= run(f'dwiextract -shell {b} "{mif}" "{shell}" -force', dry)
        ok &= run(f'mrcat "{b0}" "{shell}" -axis 3 "{cat}" -force', dry)
        ok &= run(f'mrconvert "{cat}" "{out_nii}" -export_grad_fsl '
                  f'"{os.path.join(d, f"bvecs_b0_{b}")}" "{os.path.join(d, f"bvals_b0_{b}")}" -force',
                  dry)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-dir", required=True, help="directory of HCP subject folders")
    ap.add_argument("--subject", action="append",
                    help="subject ID (repeatable); default: every folder in --base-dir")
    ap.add_argument("--shells", type=int, nargs="+", default=[1000, 2000, 3000])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo shells already extracted")
    args = ap.parse_args()

    if not os.path.isdir(args.base_dir):
        ap.error(f"not a directory: {args.base_dir}")
    subs = args.subject or sorted(s for s in os.listdir(args.base_dir)
                                  if os.path.isdir(os.path.join(args.base_dir, s)))
    failed = [s for s in subs
              if not process_subject(args.base_dir, s, args.shells, args.dry_run, args.force)]
    print(f"\n{len(subs) - len(failed)}/{len(subs)} subjects processed"
          + (f"; failed: {' '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
