import torch
import torch.nn as nn
import torch.nn.functional as F

class KPredictor(nn.Module):
    """
    Feature-Conditioned Adaptive Iteration Predictor.
    
    Predicts the optimal number of refinement iterations (K) based on
    the global spatial features (z) from the ViT backbone and the noise level (SNR).
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 128, k_max: int = 50):
        super().__init__()
        self.k_max = k_max
        
        # The input dimension is exactly embed_dim (for z) + 1 (for SNR scalar)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, vit_features: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vit_features: (B, C, H, W) features directly from the ViT encoder.
            snr: (B, 1) or (B,) tensor containing the true or estimated SNR.
            
        Returns:
            (B,) tensor containing the unrounded K predictions.
        """
        B = vit_features.shape[0]
        
        # 1. Global Spatial Pooling to create semantic feature latent 'z'
        # AdaptiveAvgPool2d(1) -> (B, C, 1, 1) -> view -> (B, C)
        z = F.adaptive_avg_pool2d(vit_features, 1).view(B, -1)
        
        # 2. Format SNR
        if snr.dim() == 1:
            snr = snr.view(B, 1)
            
        # Normalize SNR by a factor (e.g. 100 max) to keep inputs roughly in [-1, 1] scale
        snr_normalized = snr / 100.0
            
        # 3. Concatenate latents: x = [z, SNR]
        x = torch.cat([z, snr_normalized], dim=-1)
        
        # 4. Predict scalar K
        k_pred = self.mlp(x).view(B)
        
        return k_pred
        
    def predict_dynamic_k(self, vit_features: torch.Tensor, snr: torch.Tensor) -> list[int]:
        """
        Helper for inference. Runs the predictor, rounds the output, and clamps to [1, K_max].
        Returns a list of integers.
        """
        k_pred = self.forward(vit_features, snr)
        
        # Clamp and round
        k_clamped = torch.clamp(torch.round(k_pred), min=1, max=self.k_max)
        
        return k_clamped.long().cpu().tolist()
