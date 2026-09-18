#!/usr/bin/env python3
"""
validate.py

Shared validation utility for both warmup and RL training phases.

Runs inference on validation subjects, computes metrics, and saves
predictions and visualizations.

Metrics:
  ✅ FW MAE
  ✅ ICVF MAE
  ✅ Combined MAE
  ✅ SSIM (FW + ICVF)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader


def _masked_mae(pred, gt, mask):
    """Compute masked MAE (numpy arrays)."""
    return float(np.mean(np.abs(pred - gt) * mask))


def _masked_ssim(pred, gt, mask, C1=0.01**2, C2=0.03**2):
    """Compute approximate SSIM on masked voxels."""
    p = pred[mask > 0.5].flatten()
    g = gt[mask > 0.5].flatten()
    if len(p) < 10:
        return 0.0
    mu_p, mu_g = p.mean(), g.mean()
    sig_p, sig_g = p.std(), g.std()
    sig_pg = np.mean((p - mu_p) * (g - mu_g))
    ssim = ((2*mu_p*mu_g + C1) * (2*sig_pg + C2)) / \
           ((mu_p**2 + mu_g**2 + C1) * (sig_p**2 + sig_g**2 + C2))
    return float(ssim)


def run_validation(model, val_loader, device, epoch, output_dir,
                   save_predictions=True, save_visualizations=True, prefix=""):
    """Run validation inference and compute metrics.

    Args:
        model: IterativeRLModel (will be set to eval mode).
        val_loader: DataLoader for validation subjects.
        device: torch.device.
        epoch: int — current epoch number.
        output_dir: str — directory for outputs.
        save_predictions: bool — save FW/ICVF predictions.
        save_visualizations: bool — save comparison images.
        prefix: str — prefix to append to directories (e.g. 'clean_').

    Returns:
        dict with validation metrics:
            'fw_mae', 'icvf_mae', 'combined_mae', 'ssim_fw', 'ssim_icvf',
            'ssim_combined'
    """
    model.eval()

    all_fw_mae = []
    all_icvf_mae = []
    all_ssim_fw = []
    all_ssim_icvf = []

    # Store one sample for visualization
    vis_sample = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            (dwi_noisy, dwi_clean, gt_fw, gt_icvf,
             mask, tissue_masks, fc_prior) = [b.to(device) for b in batch]

            # Forward pass (no RL actions during validation)
            output = model(dwi_noisy, tissue_masks, rl_actions=None)

            fw_pred = torch.clamp(output["fw_final"], 0.0, 1.0)
            icvf_pred = torch.clamp(output["icvf_final"], 0.0, 1.0)

            # Convert to numpy for metrics
            fw_np = fw_pred.squeeze().cpu().numpy()
            icvf_np = icvf_pred.squeeze().cpu().numpy()
            gt_fw_np = gt_fw.squeeze().cpu().numpy()
            gt_icvf_np = gt_icvf.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()

            # Compute metrics
            fw_mae = _masked_mae(fw_np, gt_fw_np, mask_np)
            icvf_mae = _masked_mae(icvf_np, gt_icvf_np, mask_np)
            ssim_fw = _masked_ssim(fw_np, gt_fw_np, mask_np)
            ssim_icvf = _masked_ssim(icvf_np, gt_icvf_np, mask_np)

            all_fw_mae.append(fw_mae)
            all_icvf_mae.append(icvf_mae)
            all_ssim_fw.append(ssim_fw)
            all_ssim_icvf.append(ssim_icvf)

            # Store first batch for visualization
            if vis_sample is None:
                vis_sample = {
                    "fw_pred": fw_np,
                    "icvf_pred": icvf_np,
                    "gt_fw": gt_fw_np,
                    "gt_icvf": gt_icvf_np,
                    "mask": mask_np,
                }

            # Free GPU memory
            del output, fw_pred, icvf_pred
            torch.cuda.empty_cache()

    # Aggregate metrics
    metrics = {
        "epoch": epoch,
        "fw_mae": float(np.mean(all_fw_mae)),
        "icvf_mae": float(np.mean(all_icvf_mae)),
        "combined_mae": float(np.mean(all_fw_mae) + np.mean(all_icvf_mae)),
        "ssim_fw": float(np.mean(all_ssim_fw)),
        "ssim_icvf": float(np.mean(all_ssim_icvf)),
        "ssim_combined": float(
            (np.mean(all_ssim_fw) + np.mean(all_ssim_icvf)) / 2.0
        ),
        "n_samples": len(all_fw_mae),
    }

    # Save visualizations
    if save_visualizations and vis_sample is not None:
        try:
            from utils.visualization import save_validation_images
            save_validation_images(
                vis_sample["fw_pred"],
                vis_sample["icvf_pred"],
                vis_sample["gt_fw"],
                vis_sample["gt_icvf"],
                vis_sample["mask"],
                epoch=epoch,
                output_dir=output_dir,
                prefix=prefix,
            )
        except Exception as e:
            print(f"  ⚠️  Visualization error: {e}")

    # Save predictions as numpy
    if save_predictions and vis_sample is not None:
        pred_dir_base = f"validation_preds_{prefix.strip('_')}" if prefix else "validation_preds"
        pred_dir = os.path.join(output_dir, pred_dir_base,
                                f"epoch_{epoch:03d}")
        os.makedirs(pred_dir, exist_ok=True)
        np.save(os.path.join(pred_dir, "fw_pred.npy"),
                vis_sample["fw_pred"])
        np.save(os.path.join(pred_dir, "icvf_pred.npy"),
                vis_sample["icvf_pred"])

    # Save metrics JSON
    metrics_dir_base = f"validation_metrics_{prefix.strip('_')}" if prefix else "validation_metrics"
    metrics_dir = os.path.join(output_dir, metrics_dir_base)
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"epoch_{epoch:03d}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
