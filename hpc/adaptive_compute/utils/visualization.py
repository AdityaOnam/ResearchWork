#!/usr/bin/env python3
"""
utils/visualization.py

Visualization utilities for validation predictions.
Saves FW/ICVF prediction vs GT comparisons and multi-view slices.
"""

import os
import numpy as np
import torch


def _to_numpy(t):
    """Convert tensor/ndarray to numpy, squeezing batch/channel dims."""
    if torch.is_tensor(t):
        t = t.detach().cpu().numpy()
    while t.ndim > 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    return np.clip(t, 0.0, 1.0)


def save_slice_comparison(pred, gt, mask, output_path, title=""):
    """Save a side-by-side prediction vs GT comparison as a numpy .npz.

    This avoids matplotlib dependency. Files can be plotted later.

    Args:
        pred: (H, W) or tensor — prediction.
        gt:   (H, W) or tensor — ground truth.
        mask: (H, W) or tensor — brain mask.
        output_path: path to save .npz file.
        title: descriptive title string.
    """
    pred_np = _to_numpy(pred)
    gt_np = _to_numpy(gt)
    mask_np = _to_numpy(mask)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        pred=pred_np,
        gt=gt_np,
        mask=mask_np,
        title=title,
    )


def save_slice_comparison_png(pred, gt, mask, output_path, title=""):
    """Save a side-by-side PNG comparison (requires matplotlib).

    Falls back to .npz if matplotlib is not available.
    """
    pred_np = _to_numpy(pred)
    gt_np = _to_numpy(gt)
    mask_np = _to_numpy(mask)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(pred_np, cmap="hot", vmin=0, vmax=1)
        axes[0].set_title("Prediction")
        axes[0].axis("off")

        axes[1].imshow(gt_np, cmap="hot", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        diff = np.abs(pred_np - gt_np) * mask_np
        axes[2].imshow(diff, cmap="jet", vmin=0, vmax=0.5)
        axes[2].set_title("Abs Difference")
        axes[2].axis("off")

        if title:
            fig.suptitle(title, fontsize=14)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    except ImportError:
        # Fallback: save as npz
        npz_path = output_path.replace(".png", ".npz")
        save_slice_comparison(pred, gt, mask, npz_path, title)


def save_multi_view(volume_slices, output_path, title=""):
    """Save axial, coronal, and sagittal mid-slice views.

    Args:
        volume_slices: dict with 'axial', 'coronal', 'sagittal' keys,
                       each containing a 2D array.
        output_path: path to save.
        title: descriptive title.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        **volume_slices,
        title=title,
    )


def save_validation_images(fw_pred, icvf_pred, gt_fw, gt_icvf, mask,
                           epoch, output_dir, prefix=""):
    """Save validation prediction images for an epoch.

    Args:
        fw_pred, icvf_pred: (H, W) or tensors — predictions.
        gt_fw, gt_icvf: (H, W) or tensors — ground truth.
        mask: (H, W) or tensor — brain mask.
        epoch: int — current epoch.
        output_dir: base output directory.
        prefix: optional prefix for directory separation (e.g. 'clean_')
    """
    vis_dir_base = f"validation_vis_{prefix.strip('_')}" if prefix else "validation_vis"
    vis_dir = os.path.join(output_dir, vis_dir_base, f"epoch_{epoch:03d}")
    os.makedirs(vis_dir, exist_ok=True)

    # FW comparison
    save_slice_comparison_png(
        fw_pred, gt_fw, mask,
        os.path.join(vis_dir, "fw_comparison.png"),
        title=f"FW Prediction vs GT — Epoch {epoch}",
    )

    # ICVF comparison
    save_slice_comparison_png(
        icvf_pred, gt_icvf, mask,
        os.path.join(vis_dir, "icvf_comparison.png"),
        title=f"ICVF Prediction vs GT — Epoch {epoch}",
    )
