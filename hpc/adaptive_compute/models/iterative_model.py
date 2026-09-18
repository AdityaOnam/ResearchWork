#!/usr/bin/env python3
"""
iterative_model.py

Core model for the Physics-Informed Coarse-to-Fine Iterative Anatomical
Reconstruction using PPO-Controlled PDE Refinement.

Pipeline per forward pass (K_outer = 8 iterations):
    1. ANN Prior  →  FW_prior, ICVF_prior
    2. Concatenate [DWI(91) + FW_prior(1) + ICVF_prior(1)] = 93 channels
    3. For k = 1 … K_outer:
        a. ViT (with skip outputs) →  global features + multi-scale skips
        b. Inter-iteration gating: blend prev_features with current
        c. DSConv + ChannelGate + SE  →  local refinement
        d. Tissue-Adaptive PDE × 2 blocks  →  physics-refined features
        e. Stronger residual skip from x0
        f. Intermediate prediction  →  FW_k, ICVF_k  (deep supervision)
        g. Denoiser  →  refined input  (RL-controlled strength)
        h. Re-concatenate  [denoised_DWI + FW_k + ICVF_k]
    4. Multi-Scale Decoder (attention-weighted)  →  final FW, ICVF maps
"""

import torch
import torch.nn as nn

from models.ann_prior import DualANNPrior
from models.vit_backbone import SimpleViT
from models.pde_block import TissueAdaptivePDEBlock, get_default_pde_args
from models.denoiser import PDEDenoiser
from models.decoder import MultiScaleDecoder, DSConvBlock, SEBlock, ChannelGate
from models.predictor import KPredictor


class IterationGate(nn.Module):
    """Learned gating mechanism for inter-iteration feature blending.

    Computes a soft gate to blend features from the previous iteration
    with the current iteration's ViT output, enabling progressive
    refinement while preserving coarse anatomical structure.

    Parameters
    ----------
    channels : int
        Feature channel count.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, prev_features: torch.Tensor,
                curr_features: torch.Tensor) -> torch.Tensor:
        """Blend previous and current features via learned gate.

        Args:
            prev_features: (B, C, H, W) from previous iteration.
            curr_features: (B, C, H, W) from current ViT output.

        Returns:
            (B, C, H, W) — gated blend.
        """
        cat = torch.cat([prev_features, curr_features], dim=1)
        g = self.gate(cat)
        return g * prev_features + (1.0 - g) * curr_features


class IterativeRLModel(nn.Module):
    """Iterative physics-aware model with RL-controlled refinement.

    Parameters
    ----------
    K_outer : int
        Number of outer refinement iterations (default 8).
    K_inner : int
        Number of inner PDE sub-steps per block (default 3).
    base_ch : int
        Feature dimension / ViT embed dim (default 128).
    num_dwi : int
        Number of DWI volumes (default 91).
    vit_depth : int
        Number of ViT transformer blocks (default 8).
    vit_heads : int
        Number of multi-head attention heads (default 16).
    vit_patch_size : int
        ViT patch embedding size (default 4).
    """

    def __init__(self,
                 K_outer: int = 8,
                 K_inner: int = 3,
                 base_ch: int = 128,
                 num_dwi: int = 91,
                 vit_depth: int = 8,
                 vit_heads: int = 16,
                 vit_patch_size: int = 4):
        super().__init__()

        self.K_outer = K_outer
        self.num_dwi = num_dwi
        self.base_ch = base_ch
        
        # ── K Predictor (Adaptive Refinement) ──
        self.k_predictor = KPredictor(embed_dim=base_ch, k_max=50)

        # in_channels = DWI + FW_prior + ICVF_prior = 93
        in_ch = num_dwi + 2

        # ── Phase 3: ANN Prior ──
        self.ann_prior = DualANNPrior(ninputs=num_dwi, hidden_layers=2)

        # ── Phase 4: ViT Backbone ──
        self.vit = SimpleViT(
            in_ch=in_ch,
            embed_dim=base_ch,
            patch_size=vit_patch_size,
            depth=vit_depth,
            num_heads=vit_heads,
        )

        # ── Inter-iteration gating ──
        self.iter_gate = IterationGate(base_ch)

        # ── Local refinement (shared across iterations) ──
        self.res1 = DSConvBlock(base_ch)
        self.res2 = DSConvBlock(base_ch)
        self.gate1 = ChannelGate(base_ch)
        self.gate2 = ChannelGate(base_ch)
        self.se1 = SEBlock(base_ch, reduction=8)
        self.se2 = SEBlock(base_ch, reduction=8)

        # ── Phase 5: Tissue-Adaptive PDE blocks (2 stages) ──
        pde_args1 = get_default_pde_args(in_chs=base_ch, out_chs=base_ch,
                                          K=K_inner)
        pde_args2 = get_default_pde_args(in_chs=base_ch, out_chs=base_ch,
                                          K=K_inner)
        self.pde1 = TissueAdaptivePDEBlock(base_ch, pde_args1)
        self.pde2 = TissueAdaptivePDEBlock(base_ch, pde_args2)

        # ── Phase 6: Denoiser ──
        self.denoiser = PDEDenoiser(
            in_channels=num_dwi,
            feature_dim=base_ch,
            K_denoise=3,
        )

        # ── Residual skip weight (learnable, for x0 skip connection) ──
        self.skip_alpha = nn.Parameter(torch.tensor(0.1))

        # ── Phase 8: Multi-Scale Decoder ──
        self.decoder = MultiScaleDecoder(
            base_ch=base_ch,
            n_refine=3,
            se_reduction=8,
        )

    # ------------------------------------------------------------------
    def _single_iteration(self,
                           x: torch.Tensor,
                           tissue_masks: torch.Tensor,
                           prev_features: torch.Tensor = None) -> tuple:
        """Run one ViT → InterGate → DSConv/Gate/SE → PDE×2 iteration.

        Args:
            x: (B, 93, H, W) — concatenated input.
            tissue_masks: (B, 3, H, W) — [WM, GM, CSF].
            prev_features: (B, base_ch, H, W) — features from previous
                iteration (None for first iteration).

        Returns:
            features: (B, base_ch, H, W)
            x0: (B, base_ch, H, W) — ViT shallow features (for skip).
            skip_features: list[(B, base_ch, H, W)] — multi-scale skips.
        """
        # ViT global feature extraction with skip outputs
        x0, skip_features = self.vit.forward_with_skips(x)  # (B, base_ch, H, W)

        # Inter-iteration feature blending
        if prev_features is not None:
            x0 = self.iter_gate(prev_features, x0)

        # Local refinement stage 1
        s0_local = self.res1(x0)
        fused1 = self.gate1(s0_local, x0)
        fused1 = self.se1(fused1)

        # PDE stage 1
        g1 = self.pde1(fused1, tissue_masks)

        # NaN guard between PDE stages
        g1 = torch.nan_to_num(g1, nan=0.0, posinf=0.0, neginf=0.0)

        # Local refinement stage 2
        s_mid = self.res2(g1)
        fused2 = self.gate2(s_mid, x0)
        fused2 = self.se2(fused2)

        # PDE stage 2
        g2 = self.pde2(fused2, tissue_masks)

        # NaN guard
        g2 = torch.nan_to_num(g2, nan=0.0, posinf=0.0, neginf=0.0)

        return g2, x0, skip_features

    # ------------------------------------------------------------------
    def forward(self,
                dwi: torch.Tensor,
                tissue_masks: torch.Tensor = None,
                rl_actions: torch.Tensor = None,
                snr: torch.Tensor = None,
                force_k_max: bool = False) -> dict:
        """Full iterative forward pass.

        Args:
            dwi: (B, 91, H, W) — input DWI signal (noisy or clean).
            tissue_masks: (B, 3, H, W) — [WM, GM, CSF] masks.
                If None, tissue-adaptive PDE falls back to unmodulated mode.
            rl_actions: (B, K_outer, 3) — per-iteration RL actions:
                [denoising_strength, noise_adapt, pde_iters].
                If None (warm-up mode), uses default denoising strength.

        Returns:
            dict with keys:
                'fw_final':       (B, 1, H, W) — final FW prediction.
                'icvf_final':     (B, 1, H, W) — final ICVF prediction.
                'fw_intermediates':   list of K (B, 1, H, W) per-iteration FW.
                'icvf_intermediates': list of K (B, 1, H, W) per-iteration ICVF.
                'all_features':   list of K (B, base_ch, H, W) feature maps.
        """
        B, C, H, W = dwi.shape
        assert C == self.num_dwi, (
            f"Expected {self.num_dwi} DWI channels, got {C}"
        )

        # ── Initial ANN Prior ──
        fw_prior, icvf_prior = self.ann_prior.predict_priors_for_batch(dwi)
        # fw_prior, icvf_prior: (B, 1, H, W)

        # ── Build initial input tensor ──
        current_dwi = dwi
        current_fw = fw_prior
        current_icvf = icvf_prior

        all_features = []
        fw_intermediates = []
        icvf_intermediates = []
        x0_first = None       # save ViT shallow features from first iteration
        skip_first = None     # save multi-scale skips from first iteration
        prev_features = None  # inter-iteration feature memory
        
        K_max_run = self.K_outer
        dynamic_K_limits = None
        k_pred_raw = None
        
        # We need the first iteration's ViT features to predict K if snr is given
        if snr is not None:
            # Run one ViT pass to get features
            x_input = torch.cat([current_dwi, current_fw, current_icvf], dim=1)
            vit_features = self.vit(x_input)
            
            # Get the differentiable raw predictions for training
            k_pred_raw = self.k_predictor(vit_features, snr)
            
            # Get the integer limits for dynamic looping
            dynamic_K_limits = self.k_predictor.predict_dynamic_k(vit_features, snr)
            
            if not force_k_max:
                K_max_run = max(dynamic_K_limits)
            else:
                K_max_run = self.K_outer

        for k in range(K_max_run):
            # Concatenate: [DWI(91) + FW_prior(1) + ICVF_prior(1)]
            x_input = torch.cat([current_dwi, current_fw, current_icvf],
                                dim=1)                         # (B, 93, H, W)

            # ── Single iteration with inter-iteration gating ──
            features, x0, skip_features = self._single_iteration(
                x_input, tissue_masks, prev_features=prev_features
            )

            # Stronger residual skip from x0_first
            if x0_first is None:
                x0_first = x0
                skip_first = skip_features
            else:
                # Weighted residual from the first iteration's ViT features
                features = features + self.skip_alpha * x0_first

            all_features.append(features)

            # Update prev_features for next iteration
            prev_features = features.detach()  # detach to limit gradient flow

            # ── Intermediate prediction (deep supervision) ──
            fw_k, icvf_k = self.decoder.intermediate(features)
            
            # Clamp and guard intermediates for numerical stability under catastrophic noise
            fw_k = torch.nan_to_num(fw_k, nan=0.0, posinf=1.0, neginf=0.0)
            fw_k = torch.clamp(fw_k, 0.0, 1.0)
            icvf_k = torch.nan_to_num(icvf_k, nan=0.0, posinf=1.0, neginf=0.0)
            icvf_k = torch.clamp(icvf_k, 0.0, 1.0)

            fw_intermediates.append(fw_k)
            icvf_intermediates.append(icvf_k)

            # ── Denoising + re-input (except last iteration) ──
            if k < self.K_outer - 1:
                # Get RL denoising strength for this iteration
                if rl_actions is not None:
                    denoise_strength = rl_actions[:, k, 0:1]   # (B, 1)
                else:
                    denoise_strength = None  # use default

                # Denoise the DWI input
                current_dwi = self.denoiser(
                    current_dwi, features, denoise_strength
                )

                # Update priors with current predictions
                current_fw = fw_k.detach()
                prev_features = features

        # ── Final Multi-Scale Decoder ──
        # Final output maps
        fw_final, icvf_final = self.decoder(
            all_features, x0_first, skip_first, lengths=dynamic_K_limits
        )

        # Final stability guard
        fw_final = torch.nan_to_num(fw_final, nan=0.0, posinf=1.0, neginf=0.0)
        fw_final = torch.clamp(fw_final, 0.0, 1.0)
        icvf_final = torch.nan_to_num(icvf_final, nan=0.0, posinf=1.0, neginf=0.0)
        icvf_final = torch.clamp(icvf_final, 0.0, 1.0)

        return {
            'fw_final': fw_final,
            'icvf_final': icvf_final,
            'fw_intermediates': fw_intermediates,
            'icvf_intermediates': icvf_intermediates,
            'all_features': all_features,
            'k_pred_raw': k_pred_raw,
        }

    # ------------------------------------------------------------------
    def load_pretrained_components(self,
                                    ann_path: str = None,
                                    vit_path: str = None,
                                    device: torch.device = None):
        """Load pretrained weights for ANN prior and/or ViT backbone.

        Args:
            ann_path: Path to pretrained ANN (original single-output).
            vit_path: Path to pretrained VIT-RXN-DIFF model checkpoint.
            device: Target device.
        """
        if device is None:
            device = next(self.parameters()).device

        if ann_path:
            self.ann_prior.load_pretrained_fw(ann_path, device)

        if vit_path:
            state = torch.load(vit_path, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]

            # Map pretrained ViT keys (prefix: in_vit → vit)
            vit_state = {}
            for k, v in state.items():
                clean_k = k.replace("module.", "")
                if clean_k.startswith("in_vit."):
                    new_k = clean_k.replace("in_vit.", "vit.")
                    vit_state[new_k] = v

            if vit_state:
                missing, unexpected = self.load_state_dict(
                    vit_state, strict=False
                )
                print(f"✅ Loaded pretrained ViT: "
                      f"{len(vit_state)} keys, "
                      f"{len(missing)} missing, "
                      f"{len(unexpected)} unexpected")
            else:
                print("⚠️  No matching ViT keys found in checkpoint")
