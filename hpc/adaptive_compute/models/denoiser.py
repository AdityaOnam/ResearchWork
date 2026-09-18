#!/usr/bin/env python3
"""
denoiser.py

PDE-based denoising stage for the iterative refinement loop.

Between refinement iterations the denoiser:
  1. Takes the noisy DWI input and current feature state.
  2. Applies a lightweight PDE smoothing step.
  3. Outputs a denoised version of the input signal that will be
     re-concatenated with updated priors for the next iteration.

Enhanced stability features for extreme noise (300%):
  ✅ NaN guards at every PDE step
  ✅ Stronger clamping for numerical safety
  ✅ Adaptive denoising strength scaling

The RL agent controls the denoising strength through an action
parameter (continuous, in [0, 1]).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PDEDenoiser(nn.Module):
    """Lightweight PDE-based denoiser for inter-iteration signal refinement.

    Architecture:
        1. Feature-guided filter: uses ViT features to modulate smoothing.
        2. Learnable diffusion-only PDE (no reaction) with RL-controlled dt.
        3. Residual connection back to original input.

    Parameters
    ----------
    in_channels : int
        Number of DWI channels in the signal (default 91).
    feature_dim : int
        Dimension of the guiding feature map (default 128, from ViT).
    K_denoise : int
        Number of denoising PDE steps (default 3, lighter than main PDE).
    """

    def __init__(self, in_channels: int = 91,
                 feature_dim: int = 128,
                 K_denoise: int = 3):
        super().__init__()
        self.K = K_denoise
        self.in_channels = in_channels

        # Feature-to-modulation: learn a per-channel smoothing weight
        # from the ViT feature map
        self.feat_to_weight = nn.Sequential(
            nn.Conv2d(feature_dim, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Learnable diffusion coefficients (spatially uniform per channel)
        self.Dx = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.5)
        self.Dy = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.5)

        # Base time step (smaller for stability under extreme noise)
        self.dt_base = nn.Parameter(torch.tensor(0.03))

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,
                dwi: torch.Tensor,
                features: torch.Tensor,
                rl_denoise_strength: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            dwi: (B, 91, H, W) — noisy or partially denoised DWI signal.
            features: (B, feature_dim, H, W) — current ViT feature map.
            rl_denoise_strength: (B, 1) or scalar — RL action in [0, 1].
                Controls how aggressively to smooth. If None, uses 0.5.

        Returns:
            (B, 91, H, W) — denoised DWI signal.
        """
        B, C, H, W = dwi.shape

        # NaN guard on input
        dwi = torch.nan_to_num(dwi, nan=0.0, posinf=1.0, neginf=0.0)

        # Feature-guided smoothing weights
        smooth_weight = self.feat_to_weight(features)        # (B, C, H, W)

        # Effective diffusion coefficients
        Dx = (torch.abs(self.Dx) + 1e-6) * smooth_weight
        Dy = (torch.abs(self.Dy) + 1e-6) * smooth_weight

        # RL-controlled time step
        if rl_denoise_strength is not None:
            if rl_denoise_strength.dim() == 2:
                # (B, 1) → (B, 1, 1, 1)
                strength = rl_denoise_strength.view(B, 1, 1, 1)
            else:
                strength = rl_denoise_strength
            dt = torch.abs(self.dt_base) * strength.clamp(0.0, 1.0)
        else:
            dt = torch.abs(self.dt_base) * 0.5

        # PDE smoothing iterations (diffusion only, no reaction)
        h = dwi
        for _ in range(self.K):
            laplacian_x = (
                torch.roll(h, 1, dims=2) - 2 * h + torch.roll(h, -1, dims=2)
            )
            laplacian_y = (
                torch.roll(h, 1, dims=3) - 2 * h + torch.roll(h, -1, dims=3)
            )
            diffusion = Dx * laplacian_x + Dy * laplacian_y

            # Limit diffusion magnitude for stability under extreme noise
            diff_max = diffusion.abs().max()
            if diff_max > 10.0:
                diffusion = diffusion * (10.0 / (diff_max + 1e-8))

            h = h + dt * diffusion

            # NaN guard + clamp for stability
            h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=0.0)
            h = torch.clamp(h, 0.0, 1.0)

        # Refinement + residual connection
        h = self.refine(h)

        # Final NaN guard
        h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=0.0)

        denoised = 0.8 * h + 0.2 * dwi   # strong but not total replacement

        denoised = torch.nan_to_num(denoised, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.clamp(denoised, 0.0, 1.0)
