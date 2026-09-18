#!/usr/bin/env python3
"""
losses.py

Unified loss function for the RL iterative refinement model.

Total loss:
    L = L_FW + λ₁·L_ICVF + λ₂·L_Sobel + λ₃·L_RL + λ_aux·L_aux

Components:
    L_FW    — MAE between predicted and ground-truth free-water maps.
    L_ICVF  — MAE between predicted and ground-truth ICVF maps.
    L_Sobel — Edge-preservation loss (Sobel gradient consistency).
    L_RL    — Negative reward from the RL agent (maximise reward).
    L_aux   — Deep-supervision loss from intermediate iteration predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Individual loss components
# ──────────────────────────────────────────────────────────────────────
def mae_loss(pred: torch.Tensor,
             target: torch.Tensor,
             mask: torch.Tensor = None) -> torch.Tensor:
    """Masked Mean Absolute Error."""
    diff = torch.abs(pred - target)
    if mask is not None:
        diff = diff * mask
    return diff.mean()


def rmse_loss(pred: torch.Tensor,
              target: torch.Tensor,
              mask: torch.Tensor = None,
              eps: float = 1e-6) -> torch.Tensor:
    """Masked Root Mean Squared Error."""
    diff = (pred - target) ** 2
    if mask is not None:
        diff = diff * mask
    return torch.sqrt(diff.mean() + eps)


def sobel_loss(pred: torch.Tensor,
               target: torch.Tensor,
               mask: torch.Tensor = None) -> torch.Tensor:
    """Sobel edge-preservation loss.

    Computes L1 distance between Sobel edge maps of prediction and target.
    """
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)

    if mask is not None:
        pred = pred * mask
        target = target * mask

    gx_p = F.conv2d(pred, kx, padding=1)
    gy_p = F.conv2d(pred, ky, padding=1)
    gx_t = F.conv2d(target, kx, padding=1)
    gy_t = F.conv2d(target, ky, padding=1)

    return torch.mean(torch.abs(gx_p - gx_t) + torch.abs(gy_p - gy_t))


def deep_supervision_loss(intermediates: list,
                          gt: torch.Tensor,
                          mask: torch.Tensor = None) -> torch.Tensor:
    """Average MAE across all intermediate predictions.

    Args:
        intermediates: list of K tensors, each (B, 1, H, W).
        gt: (B, 1, H, W) ground truth.
        mask: (B, 1, H, W) optional brain mask.
    """
    if len(intermediates) == 0:
        return torch.tensor(0.0, device=gt.device)

    total = torch.tensor(0.0, device=gt.device)
    for pred_k in intermediates:
        total = total + mae_loss(pred_k, gt, mask)
    return total / len(intermediates)


# ──────────────────────────────────────────────────────────────────────
# Unified loss
# ──────────────────────────────────────────────────────────────────────
class IterativeRefinementLoss(nn.Module):
    """Composite loss for the RL iterative model.

    Parameters
    ----------
    lambda_fw : float
        Weight for FW MAE loss (default 1.0).
    lambda_icvf : float
        Weight for ICVF MAE loss (default 1.0).
    lambda_sobel : float
        Weight for Sobel edge loss (default 0.1).
    lambda_rl : float
        Weight for RL reward loss (default 0.05).
    lambda_aux : float
        Weight for deep-supervision auxiliary loss (default 0.3).
    """

    def __init__(self,
                 lambda_fw: float = 1.0,
                 lambda_icvf: float = 1.0,
                 lambda_sobel: float = 0.1,
                 lambda_rl: float = 0.05,
                 lambda_aux: float = 0.3):
        super().__init__()
        self.lambda_fw = lambda_fw
        self.lambda_icvf = lambda_icvf
        self.lambda_sobel = lambda_sobel
        self.lambda_rl = lambda_rl
        self.lambda_aux = lambda_aux

    def forward(self,
                model_output: dict,
                gt_fw: torch.Tensor,
                gt_icvf: torch.Tensor,
                mask: torch.Tensor,
                reward: torch.Tensor = None) -> tuple:
        """Compute total loss.

        Args:
            model_output: dict from IterativeRLModel.forward() with keys:
                'fw_final', 'icvf_final',
                'fw_intermediates', 'icvf_intermediates'.
            gt_fw: (B, 1, H, W) FW ground truth.
            gt_icvf: (B, 1, H, W) ICVF ground truth.
            mask: (B, 1, H, W) brain mask.
            reward: (B,) RL reward (optional, None during warm-up).

        Returns:
            total_loss: scalar tensor.
            loss_dict: dict with individual components for logging.
        """
        fw_pred = model_output["fw_final"]
        icvf_pred = model_output["icvf_final"]
        fw_inters = model_output["fw_intermediates"]
        icvf_inters = model_output["icvf_intermediates"]

        # ── FW loss ──
        l_fw = mae_loss(fw_pred, gt_fw, mask)

        # ── ICVF loss ──
        l_icvf = mae_loss(icvf_pred, gt_icvf, mask)

        # ── Sobel edge loss (on both outputs) ──
        l_sobel = (
            sobel_loss(fw_pred, gt_fw, mask)
            + sobel_loss(icvf_pred, gt_icvf, mask)
        )

        # ── Deep supervision ──
        l_aux_fw = deep_supervision_loss(fw_inters, gt_fw, mask)
        l_aux_icvf = deep_supervision_loss(icvf_inters, gt_icvf, mask)
        l_aux = l_aux_fw + l_aux_icvf

        # ── RL reward loss (negative reward → minimise) ──
        if reward is not None:
            l_rl = -reward.mean()
        else:
            l_rl = torch.tensor(0.0, device=fw_pred.device)

        # ── Total ──
        total = (
            self.lambda_fw * l_fw
            + self.lambda_icvf * l_icvf
            + self.lambda_sobel * l_sobel
            + self.lambda_rl * l_rl
            + self.lambda_aux * l_aux
        )

        loss_dict = {
            "total": total.item(),
            "fw_mae": l_fw.item(),
            "icvf_mae": l_icvf.item(),
            "sobel": l_sobel.item(),
            "rl": l_rl.item(),
            "aux": l_aux.item(),
        }

        return total, loss_dict
