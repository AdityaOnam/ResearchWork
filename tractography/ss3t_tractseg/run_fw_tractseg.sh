#!/bin/bash
# ============================================================
# FREE-WATER-CORRECTED TRACTOGRAPHY + TRACTSEG BUNDLE SEGMENTATION
#
# Input : raw DWI + a predicted free-water map (e.g. from QSSM / A3)
# Steps : FW signal subtraction (mrcalc) -> dhollander response ->
#         SS3T-CSD FODs -> whole-brain tckgen -> TDI ->
#         sh2peaks -> TractSeg (bundle / endings / TOM) -> Tracking
#
# Configure with environment variables (all optional except the data roots):
#   DWI_BASE      <DWI_BASE>/<sid>/T1w/Diffusion/dwi_b0_1000.nii.gz, bvals_b0_1000, bvecs_b0_1000
#   FW_BASE       <FW_BASE>/<sid>/predicted_free_water_volume.nii.gz
#   RESULTS_BASE  output root (one folder per subject)
#   MRTRIX_BIN, SS3T_BIN, TRACTSEG_BIN   tool locations (default: already on PATH)
#   PY            python with nibabel + numpy + dipy     (default: python3)
#   PY310         python <= 3.11 for ss3t_csd_beta1, whose shebang uses the
#                 `imp` module removed in 3.12            (default: python3)
#
#   bash run_fw_tractseg.sh                 # all IDs in $SUBJECTS_FILE
#   bash run_fw_tractseg.sh SUBJ01 SUBJ02   # explicit subjects
# ============================================================
set -uo pipefail

MRTRIX_BIN="${MRTRIX_BIN:-}"
SS3T_BIN="${SS3T_BIN:-$(dirname "$(command -v ss3t_csd_beta1 2>/dev/null || echo ./ss3t_csd_beta1)")}"
TRACTSEG_BIN="${TRACTSEG_BIN:-}"
PY="${PY:-python3}"        # has nibabel + numpy + dipy
PY310="${PY310:-python3}"  # pre-3.12, has `imp` -- for ss3t_csd_beta1
export PATH="${MRTRIX_BIN:+${MRTRIX_BIN}:}${TRACTSEG_BIN:+${TRACTSEG_BIN}:}${PATH}"
export MRTRIX_QUIET=1

SUBJECTS_FILE="${SUBJECTS_FILE:-${QSSM_SPLITS_DIR:-$(dirname "$0")/../../splits}/test.txt}"
if [ "$#" -gt 0 ]; then SUBJECTS=("$@")
elif [ -f "${SUBJECTS_FILE}" ]; then mapfile -t SUBJECTS < <(sed 's/#.*//; /^[[:space:]]*$/d' "${SUBJECTS_FILE}")
else echo "No subjects: pass IDs as arguments or create ${SUBJECTS_FILE}" >&2; exit 1; fi

: "${DWI_BASE:?set DWI_BASE to the DWI root}"
: "${FW_BASE:?set FW_BASE to the predicted free-water root}"
RESULTS_BASE="${RESULTS_BASE:-./tractography_results}"
NTHREADS="${NTHREADS:-8}"
B_VALUE=1000
FREE_WATER_DIFFUSIVITY=0.003
FW_THRESHOLD=0.7
CUTOFF=0.04
ANGLE=45
MINLENGTH=10
MAXLENGTH=250
NR_FIBERS=5000
SELECT_COUNT=200000
EXP_FACTOR=$("${PY}" -c "import math; print(math.exp(-${B_VALUE} * ${FREE_WATER_DIFFUSIVITY}))")
echo "exp(-b*D_FW) = ${EXP_FACTOR}"

LOG_DIR="${RESULTS_BASE}/pipeline_logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
FAILED_SUBJECTS=(); DONE_SUBJECTS=()
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${MASTER_LOG}"; }
run_step() { local n="$1"; shift; if ! "$@"; then log "  [STEP FAILED] ${n}"; return 1; fi; }

run_subject() {
  local SUBJ="$1"
  local SUBJ_LOG="${LOG_DIR}/${SUBJ}.log"
  local DWI="${DWI_BASE}/${SUBJ}/T1w/Diffusion/dwi_b0_1000.nii.gz"
  local BVALS="${DWI_BASE}/${SUBJ}/T1w/Diffusion/bvals_b0_1000"
  local BVECS="${DWI_BASE}/${SUBJ}/T1w/Diffusion/bvecs_b0_1000"
  local FW_FRACTION="${FW_BASE}/${SUBJ}/predicted_free_water_volume.nii.gz"
  local OUT_DIR="${RESULTS_BASE}/${SUBJ}/tractography"
  local BUNDLE_DIR="${RESULTS_BASE}/${SUBJ}/bundle_segmentation"
  mkdir -p "${OUT_DIR}" "${BUNDLE_DIR}"

  for f in "${DWI}" "${BVALS}" "${BVECS}" "${FW_FRACTION}"; do
    [ -f "${f}" ] || { log "  ERROR: Missing: ${f}"; return 1; }
  done
  [ -f "${BUNDLE_DIR}/stats/bundle_statistics.csv" ] && { log "  SKIP: ${SUBJ}"; return 0; }

  exec 3>&1 4>&2; exec >> "${SUBJ_LOG}" 2>&1

  log "  [${SUBJ}] Step 0: mrconvert"
  run_step "Step 0" mrconvert "${DWI}" -fslgrad "${BVECS}" "${BVALS}" \
    "${OUT_DIR}/dwi.mif" -nthreads ${NTHREADS} -force -quiet \
    || { exec 1>&3 2>&4; return 1; }

  log "  [${SUBJ}] Step 1: b0 -> S0 mean"
  run_step "Step 1a" dwiextract "${OUT_DIR}/dwi.mif" -bzero "${OUT_DIR}/b0_volumes.mif" -force -quiet \
    || { exec 1>&3 2>&4; return 1; }
  run_step "Step 1b" mrmath "${OUT_DIR}/b0_volumes.mif" mean -axis 3 \
    "${OUT_DIR}/S0_mean.nii.gz" -force -quiet || { exec 1>&3 2>&4; return 1; }

  log "  [${SUBJ}] Step 2: brain mask"
  run_step "Step 2" dwi2mask "${OUT_DIR}/dwi.mif" "${OUT_DIR}/brain_mask.nii.gz" -force -quiet \
    || { exec 1>&3 2>&4; return 1; }

  log "  [${SUBJ}] Step 3: FW fraction"
  run_step "Step 3a" mrcalc "${FW_FRACTION}" 0.0 -max 1.0 -min \
    "${OUT_DIR}/fw_fraction_clamped.nii.gz" -force -quiet || { exec 1>&3 2>&4; return 1; }
  run_step "Step 3b" mrcalc "${OUT_DIR}/fw_fraction_clamped.nii.gz" \
    "${OUT_DIR}/brain_mask.nii.gz" -mult \
    "${OUT_DIR}/fw_fraction_masked.nii.gz" -force -quiet || { exec 1>&3 2>&4; return 1; }
  run_step "Step 3c" mrcalc "${OUT_DIR}/fw_fraction_masked.nii.gz" ${FW_THRESHOLD} -le \
    "${OUT_DIR}/fw_threshold_mask.nii.gz" -force -quiet || { exec 1>&3 2>&4; return 1; }

  log "  [${SUBJ}] Step 4: FW-corrected tissue signal"
  run_step "Step 4a" mrcalc "${OUT_DIR}/S0_mean.nii.gz" \
    "${OUT_DIR}/fw_fraction_masked.nii.gz" -mult "${EXP_FACTOR}" -mult \
    "${OUT_DIR}/water_signal_3d.nii.gz" -force -quiet || { exec 1>&3 2>&4; return 1; }

  "${PY}" << PYEOF || { exec 1>&3 2>&4; return 1; }
import nibabel as nib, numpy as np, sys
try:
    ws = nib.load("${OUT_DIR}/water_signal_3d.nii.gz").get_fdata()
    ref = nib.load("${DWI}")
    ws4 = np.repeat(ws[..., np.newaxis], ref.shape[3], axis=3)
    nib.save(nib.Nifti1Image(ws4.astype(np.float32), ref.affine, ref.header),
             "${OUT_DIR}/water_signal_4d.nii.gz")
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)
PYEOF

  run_step "Step 4b" mrcalc "${OUT_DIR}/dwi.mif" "${OUT_DIR}/water_signal_4d.nii.gz" -subtract \
    "${OUT_DIR}/dwi_numerator.mif" -force -quiet || { exec 1>&3 2>&4; return 1; }
  run_step "Step 4c" mrcalc 1.0 "${OUT_DIR}/fw_fraction_masked.nii.gz" -subtract 0.000001 -max \
    "${OUT_DIR}/denominator.nii.gz" -force -quiet || { exec 1>&3 2>&4; return 1; }
  run_step "Step 4d" mrcalc "${OUT_DIR}/dwi_numerator.mif" "${OUT_DIR}/denominator.nii.gz" -divide \
    0.0 -max "${OUT_DIR}/fw_threshold_mask.nii.gz" -mult \
    "${OUT_DIR}/dwi_tissue.mif" -force -quiet || { exec 1>&3 2>&4; return 1; }

  log "  [${SUBJ}] Step 5: Response (dhollander)"
  run_step "Step 5" dwi2response dhollander "${OUT_DIR}/dwi_tissue.mif" \
    "${OUT_DIR}/response_wm.txt" "${OUT_DIR}/response_gm.txt" "${OUT_DIR}/response_csf.txt" \
    -mask "${OUT_DIR}/brain_mask.nii.gz" -nthreads ${NTHREADS} -force -quiet \
    || { exec 1>&3 2>&4; return 1; }

  log "  [${SUBJ}] Step 6: SS3T-CSD"
  run_step "Step 6" "${PY310}" "${SS3T_BIN}/ss3t_csd_beta1" "${OUT_DIR}/dwi_tissue.mif" \
    "${OUT_DIR}/response_wm.txt" "${OUT_DIR}/wm_fod.mif" \
    "${OUT_DIR}/response_gm.txt" "${OUT_DIR}/gm.mif" \
    "${OUT_DIR}/response_csf.txt" "${OUT_DIR}/csf.mif" \
    -mask "${OUT_DIR}/brain_mask.nii.gz" -force -quiet || { exec 1>&3 2>&4; return 1; }

  log "  [${SUBJ}] Step 7: mtnormalise"
  if mtnormalise "${OUT_DIR}/wm_fod.mif" "${OUT_DIR}/wm_fod_norm.mif" \
      "${OUT_DIR}/gm.mif" "${OUT_DIR}/gm_norm.mif" \
      -mask "${OUT_DIR}/brain_mask.nii.gz" -force -quiet; then
    log "  [${SUBJ}] mtnormalise OK"
  else
    log "  [${SUBJ}] mtnormalise FAILED — fallback"
    cp "${OUT_DIR}/wm_fod.mif" "${OUT_DIR}/wm_fod_norm.mif"
  fi

  log "  [${SUBJ}] Step 8: iFOD2 tractography"
  run_step "Step 8" tckgen "${OUT_DIR}/wm_fod_norm.mif" "${OUT_DIR}/tracks_200k.tck" \
    -algorithm iFOD2 -seed_image "${OUT_DIR}/brain_mask.nii.gz" \
    -mask "${OUT_DIR}/brain_mask.nii.gz" -select ${SELECT_COUNT} \
    -cutoff ${CUTOFF} -angle ${ANGLE} -minlength ${MINLENGTH} -maxlength ${MAXLENGTH} \
    -step 0.5 -nthreads ${NTHREADS} -force -quiet || { exec 1>&3 2>&4; return 1; }

  log "  [${SUBJ}] Step 9: TDI"
  run_step "Step 9" tckmap "${OUT_DIR}/tracks_200k.tck" "${OUT_DIR}/tdi.nii.gz" \
    -template "${OUT_DIR}/S0_mean.nii.gz" -force -quiet || { exec 1>&3 2>&4; return 1; }

  log "  [${SUBJ}] Step 10: sh2peaks"
  run_step "Step 10" sh2peaks "${OUT_DIR}/wm_fod_norm.mif" "${BUNDLE_DIR}/peaks.nii.gz" \
    -num 3 -mask "${OUT_DIR}/brain_mask.nii.gz" -nthreads ${NTHREADS} -force -quiet \
    || { exec 1>&3 2>&4; return 1; }

  log "  [${SUBJ}] Step 11: TractSeg"
  run_step "Step 11a" TractSeg -i "${BUNDLE_DIR}/peaks.nii.gz" \
    -o "${BUNDLE_DIR}/tractseg_output" --output_type tract_segmentation --nr_cpus ${NTHREADS} \
    || { exec 1>&3 2>&4; return 1; }
  run_step "Step 11b" TractSeg -i "${BUNDLE_DIR}/peaks.nii.gz" \
    -o "${BUNDLE_DIR}/tractseg_output" --output_type endings_segmentation --nr_cpus ${NTHREADS} \
    || { exec 1>&3 2>&4; return 1; }
  run_step "Step 11c" TractSeg -i "${BUNDLE_DIR}/peaks.nii.gz" \
    -o "${BUNDLE_DIR}/tractseg_output" --output_type TOM --nr_cpus ${NTHREADS} \
    || { exec 1>&3 2>&4; return 1; }

  log "  [${SUBJ}] Step 12: Tracking"
  run_step "Step 12" Tracking -i "${BUNDLE_DIR}/peaks.nii.gz" \
    -o "${BUNDLE_DIR}/tractseg_output" --algorithm prob \
    --nr_fibers ${NR_FIBERS} --tracking_format tck --nr_cpus ${NTHREADS} \
    || { exec 1>&3 2>&4; return 1; }

  log "  [${SUBJ}] Step 13: Bundle statistics"
  local STATS_DIR="${BUNDLE_DIR}/stats"; mkdir -p "${STATS_DIR}"
  "${PY}" << PYEOF || { exec 1>&3 2>&4; return 1; }
import os, glob, csv, sys, numpy as np, nibabel as nib
try:
    from dipy.io.streamline import load_tractogram
    from dipy.tracking.utils import length as streamline_length
    HAS_DIPY = True
except ImportError:
    HAS_DIPY = False
rows = []
for tck_path in sorted(glob.glob(os.path.join("${BUNDLE_DIR}/tractseg_output/TOM_trackings", "*.tck"))):
    bundle = os.path.splitext(os.path.basename(tck_path))[0]
    mask_path = os.path.join("${BUNDLE_DIR}/tractseg_output/bundle_segmentations", bundle + ".nii.gz")
    n_sl = 0; mean_l = 0.0; med_l = 0.0; vol = 0.0
    if HAS_DIPY:
        try:
            sft = load_tractogram(tck_path, "${OUT_DIR}/S0_mean.nii.gz", bbox_valid_check=False)
            sft.to_vox(); n_sl = len(sft.streamlines)
            if n_sl > 0:
                lens = list(streamline_length(sft.streamlines))
                mean_l = float(np.mean(lens)); med_l = float(np.median(lens))
        except Exception as e:
            print(f"  WARNING {bundle}: {e}", file=sys.stderr)
    if os.path.exists(mask_path):
        img = nib.load(mask_path)
        vol = float(np.sum(img.get_fdata() > 0.5) * np.prod(img.header.get_zooms()[:3]))
    rows.append({"bundle": bundle, "n_streamlines": n_sl,
                 "mean_length_mm": round(mean_l,2), "median_length_mm": round(med_l,2), "volume_mm3": round(vol,2)})
if not rows:
    print("  WARNING: No .tck files found", file=sys.stderr)
csv_path = os.path.join("${STATS_DIR}", "bundle_statistics.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["bundle","n_streamlines","mean_length_mm","median_length_mm","volume_mm3"])
    w.writeheader(); w.writerows(rows)
print(f"  Saved: {csv_path}  ({len(rows)} bundles)")
PYEOF

  exec 1>&3 2>&4; log "  [${SUBJ}] DONE"; return 0
}

TOTAL=${#SUBJECTS[@]}; COUNT=0
log "Starting Pred-FW-A3 HCP (local): ${TOTAL} subjects"
START_TIME=$(date +%s)
for SUBJ in "${SUBJECTS[@]}"; do
  COUNT=$((COUNT+1)); log "Progress: ${COUNT}/${TOTAL} — ${SUBJ}"
  if run_subject "${SUBJ}"; then DONE_SUBJECTS+=("${SUBJ}")
  else log "  FAILED: ${SUBJ}"; FAILED_SUBJECTS+=("${SUBJ}"); fi
done
ELAPSED=$(( ($(date +%s) - START_TIME) / 60 ))
log "COMPLETE — Done: ${#DONE_SUBJECTS[@]}  Failed: ${#FAILED_SUBJECTS[@]}  Time: ${ELAPSED}m"
