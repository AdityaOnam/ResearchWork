#!/usr/bin/env python3
"""
ann_prior.py

Dual-output ANN prior for joint FW and ICVF prediction from voxel-wise DWI.

Architecture (shared backbone, dual heads):
    91 → 91 (input) → 136 (hidden1) → 68 (hidden2) → 2 (FW + ICVF, sigmoid)

This extends the existing ANN from the Stage-I voxel-wise ANN free-water estimator
to produce two outputs while preserving the original layer sizing logic:
    hidden1 = int((ninputs / hidden_layers) * 1.5)
    hidden2 = int((ninputs / (hidden_layers * 2)) * 1.5)
"""

import torch
import torch.nn as nn
import numpy as np


class DualANNPrior(nn.Module):
    """ANN that predicts both FW fraction and ICVF from normalised DWI signal.

    Parameters
    ----------
    ninputs : int
        Number of input features (= number of b-values / DWI volumes).
        Default 91 for b0 + 90 directions.
    hidden_layers : int
        Controls hidden layer widths (matches the original ANN sizing).
        Default 2 (matching unique_bvals.size for single-shell b=1000).
    """

    def __init__(self, ninputs: int = 91, hidden_layers: int = 2):
        super().__init__()
        self.ninputs = ninputs
        self.hidden_layers = hidden_layers

        h1 = int((ninputs / hidden_layers) * 1.5)       # 136 for ninputs=91
        h2 = int((ninputs / (hidden_layers * 2)) * 1.5)  # 68  for ninputs=91

        # Shared backbone (identical to existing ANN)
        self.backbone = nn.Sequential(
            nn.Linear(ninputs, ninputs),
            nn.ReLU(),
            nn.Linear(ninputs, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )

        # Separate output heads
        self.fw_head = nn.Sequential(
            nn.Linear(h2, 1),
            nn.Sigmoid(),
        )
        self.icvf_head = nn.Sequential(
            nn.Linear(h2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, ninputs) normalised DWI signal per voxel.

        Returns:
            fw:   (B, 1) predicted free-water fraction.
            icvf: (B, 1) predicted intracellular volume fraction.
        """
        features = self.backbone(x)
        fw = self.fw_head(features)
        icvf = self.icvf_head(features)
        return fw, icvf

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def load_pretrained_fw(self, path: str, device: torch.device = None):
        """Load pretrained FW-only weights from original ANN checkpoint.

        The original ANN has:
            input_layer, hidden_layer1, hidden_layer2, output_layer
        We map these to:
            backbone[0], backbone[2], backbone[4], fw_head[0]
        """
        if device is None:
            device = next(self.parameters()).device

        state = torch.load(path, map_location=device)

        mapping = {
            "input_layer.weight":   "backbone.0.weight",
            "input_layer.bias":     "backbone.0.bias",
            "hidden_layer1.weight": "backbone.2.weight",
            "hidden_layer1.bias":   "backbone.2.bias",
            "hidden_layer2.weight": "backbone.4.weight",
            "hidden_layer2.bias":   "backbone.4.bias",
            "output_layer.weight":  "fw_head.0.weight",
            "output_layer.bias":    "fw_head.0.bias",
        }

        new_state = {}
        for old_key, new_key in mapping.items():
            if old_key in state:
                new_state[new_key] = state[old_key]

        missing, unexpected = self.load_state_dict(new_state, strict=False)
        print(f"✅ Loaded pretrained FW weights: "
              f"{len(new_state)} keys mapped, "
              f"{len(missing)} missing (ICVF head expected)")

    # ------------------------------------------------------------------
    def predict_priors_for_slice(self,
                                  dwi_slice: torch.Tensor) -> tuple:
        """Predict FW and ICVF prior maps for a full 2-D slice.

        Args:
            dwi_slice: (C, H, W) normalised DWI slice with C=91 channels.

        Returns:
            fw_map:   (1, H, W) FW prior map.
            icvf_map: (1, H, W) ICVF prior map.
        """
        C, H, W = dwi_slice.shape
        # Reshape to (H*W, C) → voxel-wise prediction
        voxels = dwi_slice.permute(1, 2, 0).reshape(-1, C)   # (H*W, C)

        with torch.no_grad():
            fw, icvf = self.forward(voxels)                   # (H*W, 1) each

        fw_map = fw.reshape(H, W).unsqueeze(0)                # (1, H, W)
        icvf_map = icvf.reshape(H, W).unsqueeze(0)
        return fw_map, icvf_map

    def predict_priors_for_batch(self,
                                  dwi_batch: torch.Tensor) -> tuple:
        """Predict FW and ICVF prior maps for a batch of slices.

        Args:
            dwi_batch: (B, C, H, W) normalised DWI batch with C=91.

        Returns:
            fw_maps:   (B, 1, H, W) FW prior maps.
            icvf_maps: (B, 1, H, W) ICVF prior maps.
        """
        B, C, H, W = dwi_batch.shape
        # Reshape to (B*H*W, C)
        voxels = dwi_batch.permute(0, 2, 3, 1).reshape(-1, C)

        fw, icvf = self.forward(voxels)

        fw_maps = fw.reshape(B, H, W).unsqueeze(1)
        icvf_maps = icvf.reshape(B, H, W).unsqueeze(1)
        return fw_maps, icvf_maps
