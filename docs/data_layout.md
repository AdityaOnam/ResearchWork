# Data layout

This repository ships no data. The code expects the directory layouts below and finds
them through environment variables ([qssm/paths.py](../qssm/paths.py)). The layouts follow
the Human Connectome Project (HCP) convention. Any cohort arranged the same way works.

## Training cohort: `$QSSM_TRAIN_SIGNAL`, `$QSSM_TRAIN_GT`

```
$QSSM_TRAIN_SIGNAL/<sid>/
    dwi.nii.gz                  (X, Y, Z, 91)   b0 + 90 directions at b = 1000 s/mm²
    bvals, bvecs                FSL format
    nodif_brain_mask.nii.gz
$QSSM_TRAIN_GT/<sid>/
    free_water_volume.nii.gz    free-water fraction ground truth (DIPY multi-shell bi-tensor)
```

Subject IDs come from `splits/train.txt`. The trainer holds out three of these subjects
for validation.

## Test cohort: `$QSSM_TEST_ROOT`

```
$QSSM_TEST_ROOT/
    Signal/<sid>/T1w/Diffusion/dwi_b0_1000.nii.gz        (+ bvals_b0_1000, bvecs_b0_1000)
    t_data/<sid>/T1w/Diffusion/nodif_brain_mask.nii.gz   (+ data.nii.gz, bvals, bvecs for FA/MD)
    Analysis/multi_shell/Ground_Truth/<sid>/
        free_water_volume.nii.gz, FA_fwe.nii.gz, MD_fwe.nii.gz, dti_FA.nii.gz, dti_MD.nii.gz
```

Subject IDs come from `splits/test.txt`.

## Stage-I prior: `$QSSM_PRIOR_ROOT`

```
$QSSM_PRIOR_ROOT/<sid>/dwi_b0_1000/fw_volume_fraction.nii.gz    (test)
$QSSM_PRIOR_ROOT/<sid>/predicted_fw_volume_fraction.nii.gz      (train, RDT-Net baseline)
```

Only the 92-channel RDT-Net baseline uses the prior. The QSSM arms take the 91 DWI
channels only.

## External cohort (optional): `$QSSM_EXT_ROOT`

```
$QSSM_EXT_ROOT/<sid>/ses-preop/dwi/resized/
    dwi_regrid.nii.gz, bvals_regrid, bvecs_regrid, dwi_regrid_mask.nii.gz
    ground_truth/free_water_volume.nii.gz, FA.nii.gz, MD.nii.gz
```

Subject IDs come from `splits/external.txt`. Used by `evaluate_fa_md.py --cohort btc`.

## Producing single-shell inputs

`dwi_b0_1000.nii.gz` (one b0 followed by the b = 1000 shell) is made from raw HCP
`data.nii.gz` by [hpc/preprocessing/shell_extractor.py](../hpc/preprocessing/shell_extractor.py).
