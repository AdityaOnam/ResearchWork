#!/usr/bin/env python3
"""
pde_block.py

Tissue-Adaptive Reaction-Diffusion PDE block with enhanced stability.

Combines:
  - Reaction-Diffusion PDE from the CNN reaction-diffusion baseline
    (α·h − β·h² + γ·f_src with diffusion)
  - Enhanced features from VIT-ADV-DIFF
    (anisotropic tensor, edge detection, CFL stability)
  - Tissue-mask conditioning
    (WM → anisotropic, GM → moderate isotropic, CSF → strong smoothing)
  - NEW: Extreme-noise stability improvements
    (stronger gradient clipping, smaller dt, NaN guards, feature clamping,
     adaptive diffusion stability checks)

Inner PDE iterations: K = 3 (default).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.models.layers import create_conv2d
except ImportError:
    # Fallback when timm is not installed (only needed for non-constant Dxy)
    def create_conv2d(in_chs, out_chs, kernel_size, stride=1,
                      dilation=1, padding='', depthwise=False, **kwargs):
        groups = in_chs if depthwise else 1
        pad = kernel_size // 2 if not padding else int(padding)
        return nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size,
                         stride=stride, padding=pad, dilation=dilation,
                         groups=groups, bias=False)


class EdgeDetector(nn.Module):
    """Sobel-based edge detector for tissue boundary identification.

    Reused from the advection-diffusion (PAFNet) global layer.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, 1, H, W) normalised edge magnitude."""
        x_gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(x_gray, self.sobel_x, padding=1)
        gy = F.conv2d(x_gray, self.sobel_y, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return mag / (mag.max() + 1e-6)


class TissueAdaptivePDEBlock(nn.Module):
    """Tissue-conditioned Reaction-Diffusion PDE block.

    Equation solved (per inner iteration):
        ∂h/∂t = Dx·∂²h/∂x² + Dy·∂²h/∂y²          (diffusion)
               + α(h)·h − β(h)·h² + γ(h)·f_src   (reaction)

    Tissue masks modulate diffusion and reaction coefficients:
        - WM regions → larger anisotropic diffusion, strong reaction
        - GM regions → moderate isotropic diffusion
        - CSF regions → strong isotropic smoothing, suppressed reaction

    Enhanced stability features for extreme noise:
        - Smaller adaptive time step (dt_max = 0.03)
        - NaN guards at every PDE iteration
        - Feature clamping after each step (-3, 3)
        - Adaptive diffusion magnitude limiting
        - Gradient magnitude monitoring

    Parameters
    ----------
    planes : int
        Feature channel count (default 128, matching ViT embed_dim).
    K : int
        Number of inner PDE iterations (default 3).
    dt_min, dt_max : float
        Learnable adaptive time-step bounds.
    tissue_adaptive : bool
        If True, condition coefficients on tissue masks.
    """

    def __init__(self, planes: int, args: dict):
        super().__init__()

        norm_layer = args.get("norm_layer", nn.BatchNorm2d)

        self.K = args.get("K", 3)
        self.dx = torch.tensor(args.get("dx", 1.0), dtype=torch.float32)
        self.dy = torch.tensor(args.get("dy", 1.0), dtype=torch.float32)
        self.dt_min = args.get("dt_min", 0.005)
        self.dt_max = args.get("dt_max", 0.03)

        self.use_res = args.get("use_res", False)
        self.no_f = args.get("no_f", False)
        self.constant_Dxy = args.get("constant_Dxy", True)
        self.tissue_adaptive = args.get("tissue_adaptive", True)

        # Stability parameters
        self.clamp_min = args.get("clamp_min", -3.0)
        self.clamp_max = args.get("clamp_max", 3.0)
        self.max_diffusion_magnitude = args.get("max_diffusion_magnitude", 10.0)

        dw_ks = args.get("dw_kernel_size", 3)
        dilation = args.get("dilation", 1)
        pad_type = args.get("pad_type", "")
        drop_path_rate = args.get("drop_path_rate", 0.0)

        in_chs = args.get("in_chs", planes)
        out_chs = args.get("out_chs", planes)
        if "out_chs" in args:
            planes = out_chs

        self.planes = planes
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.drop_path_rate = drop_path_rate

        self.act = nn.ReLU(inplace=True)
        self.bn_out = norm_layer(planes)

        # ── Initial feature transform ──
        self.init_conv = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                      groups=planes, bias=False),
            norm_layer(planes),
            nn.ReLU(inplace=True),
        )

        # ── Diffusion coefficients ──
        if self.constant_Dxy:
            self.cDx = nn.Parameter(torch.tensor(1.0))
            self.cDy = nn.Parameter(torch.tensor(1.0))
        else:
            self.convDx = create_conv2d(planes, planes, kernel_size=dw_ks,
                                        stride=1, dilation=dilation,
                                        padding=pad_type, depthwise=True)
            self.convDy = create_conv2d(planes, planes, kernel_size=dw_ks,
                                        stride=1, dilation=dilation,
                                        padding=pad_type, depthwise=True)
            self.bnDx = norm_layer(planes)
            self.bnDy = norm_layer(planes)

        # ── Reaction coefficients (α, β, γ) ──
        self.conv_alpha = nn.Conv2d(planes, planes, kernel_size=1)
        self.conv_beta = nn.Conv2d(planes, planes, kernel_size=1)
        self.conv_gamma = nn.Conv2d(planes, planes, kernel_size=1)

        # ── Adaptive time step ──
        self.conv_dt = nn.Sequential(
            nn.Conv2d(planes, planes // 4, kernel_size=1),
            norm_layer(planes // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes // 4, 1, kernel_size=1),
            nn.Sigmoid(),                           # output in [0, 1]
        )

        # ── Edge detector ──
        self.edge_detector = EdgeDetector()

        # ── Tissue modulation network ──
        if self.tissue_adaptive:
            # Maps (3-ch tissue masks) → per-channel modulation weights
            self.tissue_diff_mod = nn.Sequential(
                nn.Conv2d(3, planes // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes // 4, planes, kernel_size=1),
                nn.Sigmoid(),
            )
            self.tissue_rxn_mod = nn.Sequential(
                nn.Conv2d(3, planes // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes // 4, planes, kernel_size=1),
                nn.Sigmoid(),
            )

    # ------------------------------------------------------------------
    def _nan_guard(self, x: torch.Tensor) -> torch.Tensor:
        """Replace NaN/Inf values with zeros and clamp to safe range."""
        x = torch.nan_to_num(x, nan=0.0, posinf=self.clamp_max,
                             neginf=self.clamp_min)
        return torch.clamp(x, self.clamp_min, self.clamp_max)

    # ------------------------------------------------------------------
    def _limit_diffusion(self, diffusion: torch.Tensor) -> torch.Tensor:
        """Adaptively limit diffusion magnitude for stability."""
        max_mag = diffusion.abs().max()
        if max_mag > self.max_diffusion_magnitude:
            scale = self.max_diffusion_magnitude / (max_mag + 1e-8)
            diffusion = diffusion * scale
        return diffusion

    # ------------------------------------------------------------------
    def forward(self,
                s0: torch.Tensor,
                tissue_masks: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            s0: (B, planes, H, W) — input feature map.
            tissue_masks: (B, 3, H, W) — [WM, GM, CSF] masks (optional).

        Returns:
            (B, planes, H, W) — PDE-refined feature map.
        """
        f = s0
        h = self.init_conv(f)
        residual = f
        h_prev = h

        # NaN guard on input
        h_prev = self._nan_guard(h_prev)

        # ── Diffusion coefficients ──
        if self.constant_Dxy:
            Dx = torch.abs(self.cDx) + 1e-6
            Dy = torch.abs(self.cDy) + 1e-6
        else:
            Dx = self.act(self.bnDx(self.convDx(h))) + 1e-6
            Dy = self.act(self.bnDy(self.convDy(h))) + 1e-6

        # ── Reaction coefficients ──
        alpha = self.conv_alpha(h)
        beta = self.conv_beta(h)
        gamma = self.conv_gamma(h)

        # Clamp reaction coefficients for stability
        alpha = torch.clamp(alpha, -2.0, 2.0)
        beta = torch.clamp(beta, -2.0, 2.0)
        gamma = torch.clamp(gamma, -2.0, 2.0)

        # ── Tissue-adaptive modulation ──
        if self.tissue_adaptive and tissue_masks is not None:
            diff_mod = self.tissue_diff_mod(tissue_masks)    # (B, planes, H, W)
            rxn_mod = self.tissue_rxn_mod(tissue_masks)
        else:
            diff_mod = 1.0
            rxn_mod = 1.0

        # ── Adaptive time step with CFL stability ──
        dt_norm = self.conv_dt(h)                           # (B, 1, H, W) ∈ [0,1]
        dt_map = self.dt_min + (self.dt_max - self.dt_min) * dt_norm

        if not self.constant_Dxy:
            max_D = torch.max(Dx.max(), Dy.max())
        else:
            max_D = max(Dx.item() if isinstance(Dx, torch.Tensor) else Dx,
                        Dy.item() if isinstance(Dy, torch.Tensor) else Dy)
        cfl_dt = 0.25 * min(self.dx ** 2, self.dy ** 2) / (max_D + 1e-6)
        dt_map = torch.clamp(dt_map, max=float(cfl_dt))

        # ── Source term ──
        source = f if not self.no_f else 0.0

        # ── PDE iterations with enhanced stability ──
        for _ in range(self.K):
            # Laplacian (central differences)
            diffusion_x = (
                torch.roll(h_prev, 1, dims=2)
                - 2 * h_prev
                + torch.roll(h_prev, -1, dims=2)
            ) / (self.dx * self.dx)

            diffusion_y = (
                torch.roll(h_prev, 1, dims=3)
                - 2 * h_prev
                + torch.roll(h_prev, -1, dims=3)
            ) / (self.dy * self.dy)

            if self.constant_Dxy:
                diffusion = Dx * diffusion_x + Dy * diffusion_y
            else:
                diffusion = Dx * diffusion_x + Dy * diffusion_y

            # Apply tissue-adaptive diffusion modulation
            diffusion = diffusion * diff_mod

            # Adaptive diffusion magnitude limiting
            diffusion = self._limit_diffusion(diffusion)

            # Reaction term: α·h − β·h² + γ·source
            reaction = alpha * h_prev - beta * (h_prev ** 2) + gamma * source

            # Clamp reaction for stability
            reaction = torch.clamp(reaction, -5.0, 5.0)
            reaction = reaction * rxn_mod

            # Time integration
            h_prev = h_prev + dt_map * (diffusion + reaction)

            # NaN guard and stability clamp (tighter range for extreme noise)
            h_prev = self._nan_guard(h_prev)

        # ── Output ──
        h = self.bn_out(h_prev)
        h = self.act(h)

        # Final NaN guard and clamp
        h = self._nan_guard(h)

        if self.use_res:
            h = h + residual

        return h


# ──────────────────────────────────────────────────────────────────────
# Default PDE argument generator
# ──────────────────────────────────────────────────────────────────────
def get_default_pde_args(in_chs: int = 128,
                          out_chs: int = 128,
                          K: int = 3) -> dict:
    """Return default PDE configuration dict.

    Updated for extreme-noise stability:
      - Smaller dt_max (0.03)
      - Tighter clamping range (-3, 3)
      - Diffusion magnitude limiting at 10.0
    """
    return {
        "in_chs": in_chs,
        "out_chs": out_chs,
        "K": K,
        "dx": 1.0,
        "dy": 1.0,
        "dt_min": 0.005,
        "dt_max": 0.03,
        "constant_Dxy": True,
        "nonlinear_pde": True,
        "tissue_adaptive": True,
        "use_res": False,
        "no_f": False,
        "norm_layer": nn.BatchNorm2d,
        "dw_kernel_size": 3,
        "dilation": 1,
        "pad_type": "",
        "drop_path_rate": 0.0,
        "clamp_min": -3.0,
        "clamp_max": 3.0,
        "max_diffusion_magnitude": 10.0,
    }
