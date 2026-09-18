#!/usr/bin/env python3
"""
reward.py

Enhanced composite reward function for the RL agent in the iterative
refinement loop.

Reward components:
  + FW prediction improvement (MAE decrease from previous iteration)
  + ICVF prediction improvement
  + SSIM reward (structural similarity)
  + Sobel edge preservation reward
  + Anatomical continuity reward (WM smoothness)
  + Inter-iteration consistency reward (progressive improvement)
  - Prediction instability (large inter-iteration jumps)
  - Artifact generation (out-of-range values, NaN)
  - Hallucination penalty (unrealistic predictions)
"""

import torch
import torch.nn.functional as F


def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge magnitude for a (B, 1, H, W) tensor."""
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


def _ssim_reward(pred: torch.Tensor, gt: torch.Tensor,
                 mask: torch.Tensor,
                 C1: float = 0.01 ** 2,
                 C2: float = 0.03 ** 2) -> torch.Tensor:
    """Compute per-sample approximate SSIM as a reward signal.

    Args:
        pred: (B, 1, H, W) prediction.
        gt:   (B, 1, H, W) ground truth.
        mask: (B, 1, H, W) brain mask.

    Returns:
        (B,) SSIM values in [-1, 1] (higher = better).
    """
    B = pred.shape[0]
    ssim_vals = torch.zeros(B, device=pred.device)

    for i in range(B):
        p = (pred[i] * mask[i]).flatten()
        g = (gt[i] * mask[i]).flatten()
        m = mask[i].flatten()

        # Only compute on brain voxels
        valid = m > 0.5
        if valid.sum() < 10:
            continue

        pv = p[valid]
        gv = g[valid]

        mu_p = pv.mean()
        mu_g = gv.mean()
        sig_p = pv.var() + 1e-8
        sig_g = gv.var() + 1e-8
        sig_pg = ((pv - mu_p) * (gv - mu_g)).mean()

        ssim = ((2 * mu_p * mu_g + C1) * (2 * sig_pg + C2)) / \
               ((mu_p ** 2 + mu_g ** 2 + C1) * (sig_p + sig_g + C2))
        ssim_vals[i] = ssim

    return ssim_vals


def _edge_preservation_reward(pred: torch.Tensor, gt: torch.Tensor,
                               mask: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge preservation reward.

    Measures how well the prediction's edge structure matches GT edges.

    Args:
        pred, gt: (B, 1, H, W)
        mask: (B, 1, H, W)

    Returns:
        (B,) edge reward (negative = penalty for mismatched edges).
    """
    pred_edges = _sobel_edges(pred * mask)
    gt_edges = _sobel_edges(gt * mask)

    # L1 distance of edge maps
    edge_diff = (torch.abs(pred_edges - gt_edges) * mask).mean(dim=(1, 2, 3))
    return -edge_diff  # negative = penalty


def _anatomical_continuity_reward(pred: torch.Tensor,
                                   mask: torch.Tensor) -> torch.Tensor:
    """Reward spatial smoothness (continuity) within the brain.

    Penalises large spatial gradients that suggest discontinuous
    (hallucinated) tissue boundaries.

    Args:
        pred: (B, 1, H, W)
        mask: (B, 1, H, W)

    Returns:
        (B,) continuity reward (negative = penalty for discontinuity).
    """
    # Spatial gradients
    grad_x = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    grad_y = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])

    # Mask the gradients
    mask_x = mask[:, :, 1:, :]
    mask_y = mask[:, :, :, 1:]

    mean_grad = (
        (grad_x * mask_x).mean(dim=(1, 2, 3))
        + (grad_y * mask_y).mean(dim=(1, 2, 3))
    )

    return -mean_grad  # negative = penalty for large gradients


def _hallucination_penalty(pred: torch.Tensor,
                            gt: torch.Tensor,
                            mask: torch.Tensor,
                            threshold: float = 0.5) -> torch.Tensor:
    """Penalise hallucinated predictions far from ground truth.

    Detects voxels where the prediction is significantly different
    from GT (beyond a threshold) — likely hallucinated anatomy.

    Args:
        pred: (B, 1, H, W)
        gt:   (B, 1, H, W)
        mask: (B, 1, H, W)
        threshold: difference threshold for hallucination detection.

    Returns:
        (B,) hallucination penalty (always ≥ 0, higher = worse).
    """
    diff = torch.abs(pred - gt)
    hallucinated = (diff > threshold).float() * mask
    halluc_fraction = hallucinated.mean(dim=(1, 2, 3))

    # Also penalise the magnitude of hallucination
    halluc_magnitude = (diff * hallucinated).mean(dim=(1, 2, 3))

    return halluc_fraction + halluc_magnitude


def _inter_iteration_consistency(pred: torch.Tensor,
                                  prev_pred: torch.Tensor,
                                  gt: torch.Tensor,
                                  mask: torch.Tensor) -> torch.Tensor:
    """Reward consistent progressive improvement across iterations.

    Penalises oscillations (prediction getting worse then better)
    and rewards smooth convergence toward GT.

    Args:
        pred: (B, 1, H, W) current prediction.
        prev_pred: (B, 1, H, W) previous iteration prediction.
        gt: (B, 1, H, W)
        mask: (B, 1, H, W)

    Returns:
        (B,) consistency reward (positive = improving, negative = oscillating).
    """
    curr_err = (torch.abs(pred - gt) * mask).mean(dim=(1, 2, 3))
    prev_err = (torch.abs(prev_pred - gt) * mask).mean(dim=(1, 2, 3))

    # Improvement (positive is good)
    improvement = prev_err - curr_err

    # Penalise direction changes (oscillation)
    direction = torch.sign(improvement)
    # Small bonus for consistent direction, penalty for oscillation
    return improvement * direction.abs()


def compute_reward(
    pred_fw: torch.Tensor,
    pred_icvf: torch.Tensor,
    gt_fw: torch.Tensor,
    gt_icvf: torch.Tensor,
    prev_pred_fw: torch.Tensor,
    prev_pred_icvf: torch.Tensor,
    mask: torch.Tensor,
    w_fw: float = 1.0,
    w_icvf: float = 1.0,
    w_ssim: float = 0.5,
    w_edge: float = 0.4,
    w_continuity: float = 0.3,
    w_hallucination: float = 0.5,
    w_consistency: float = 0.3,
    w_instability: float = 0.3,
    w_artifact: float = 0.2,
) -> tuple:
    """Compute enhanced composite reward for a single refinement step.

    All inputs are (B, 1, H, W) tensors.

    Args:
        pred_fw:       Current FW prediction.
        pred_icvf:     Current ICVF prediction.
        gt_fw:         FW ground truth.
        gt_icvf:       ICVF ground truth.
        prev_pred_fw:  FW prediction from previous iteration.
        prev_pred_icvf: ICVF prediction from previous iteration.
        mask:          Brain mask (1 = brain, 0 = background).
        w_*:           Reward component weights.

    Returns:
        reward: (B,) per-sample scalar reward.
        info:   dict with individual reward components (for logging).
    """
    eps = 1e-8

    # ── 1. MAE Improvement reward (positive if prediction gets closer to GT) ──
    prev_fw_mae = (torch.abs(prev_pred_fw - gt_fw) * mask).mean(dim=(1, 2, 3))
    curr_fw_mae = (torch.abs(pred_fw - gt_fw) * mask).mean(dim=(1, 2, 3))
    fw_improvement = prev_fw_mae - curr_fw_mae   # positive = good

    prev_icvf_mae = (torch.abs(prev_pred_icvf - gt_icvf) * mask).mean(dim=(1, 2, 3))
    curr_icvf_mae = (torch.abs(pred_icvf - gt_icvf) * mask).mean(dim=(1, 2, 3))
    icvf_improvement = prev_icvf_mae - curr_icvf_mae

    # ── 2. SSIM reward ──
    ssim_fw = _ssim_reward(pred_fw, gt_fw, mask)
    ssim_icvf = _ssim_reward(pred_icvf, gt_icvf, mask)
    ssim_reward = (ssim_fw + ssim_icvf) / 2.0

    # ── 3. Sobel edge preservation reward ──
    edge_fw = _edge_preservation_reward(pred_fw, gt_fw, mask)
    edge_icvf = _edge_preservation_reward(pred_icvf, gt_icvf, mask)
    edge_reward = (edge_fw + edge_icvf) / 2.0

    # ── 4. Anatomical continuity reward ──
    cont_fw = _anatomical_continuity_reward(pred_fw, mask)
    cont_icvf = _anatomical_continuity_reward(pred_icvf, mask)
    continuity_reward = (cont_fw + cont_icvf) / 2.0

    # ── 5. Hallucination penalty ──
    halluc_fw = _hallucination_penalty(pred_fw, gt_fw, mask)
    halluc_icvf = _hallucination_penalty(pred_icvf, gt_icvf, mask)
    hallucination_pen = (halluc_fw + halluc_icvf) / 2.0

    # ── 6. Inter-iteration consistency ──
    consist_fw = _inter_iteration_consistency(
        pred_fw, prev_pred_fw, gt_fw, mask
    )
    consist_icvf = _inter_iteration_consistency(
        pred_icvf, prev_pred_icvf, gt_icvf, mask
    )
    consistency_reward = (consist_fw + consist_icvf) / 2.0

    # ── 7. Prediction instability (penalise large jumps) ──
    fw_jump = (torch.abs(pred_fw - prev_pred_fw) * mask).mean(dim=(1, 2, 3))
    icvf_jump = (torch.abs(pred_icvf - prev_pred_icvf) * mask).mean(dim=(1, 2, 3))
    instability = fw_jump + icvf_jump

    # ── 8. Artifact penalty (out-of-range or NaN) ──
    out_of_range_fw = (
        (pred_fw < 0).float() + (pred_fw > 1).float()
    ).mean(dim=(1, 2, 3))
    out_of_range_icvf = (
        (pred_icvf < 0).float() + (pred_icvf > 1).float()
    ).mean(dim=(1, 2, 3))
    nan_penalty = (
        torch.isnan(pred_fw).float().mean(dim=(1, 2, 3))
        + torch.isnan(pred_icvf).float().mean(dim=(1, 2, 3))
    )
    artifact_penalty = out_of_range_fw + out_of_range_icvf + nan_penalty

    # ── Composite reward ──
    reward = (
        w_fw * fw_improvement
        + w_icvf * icvf_improvement
        + w_ssim * ssim_reward
        + w_edge * edge_reward
        + w_continuity * continuity_reward
        + w_consistency * consistency_reward
        - w_hallucination * hallucination_pen
        - w_instability * instability
        - w_artifact * artifact_penalty
    )

    info = {
        "fw_improvement": fw_improvement.mean().item(),
        "icvf_improvement": icvf_improvement.mean().item(),
        "ssim_reward": ssim_reward.mean().item(),
        "edge_reward": edge_reward.mean().item(),
        "continuity_reward": continuity_reward.mean().item(),
        "hallucination_penalty": hallucination_pen.mean().item(),
        "consistency_reward": consistency_reward.mean().item(),
        "instability": instability.mean().item(),
        "artifact_penalty": artifact_penalty.mean().item(),
        "reward_mean": reward.mean().item(),
    }

    return reward, info
