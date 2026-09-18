import torch
import numpy as np
from dipy.denoise.noise_estimate import piesno

def estimate_snr_piesno(dwi_tensor: torch.Tensor, N: int = 1) -> torch.Tensor:
    """
    Estimates the Signal-to-Noise Ratio (SNR) using the PIESNO algorithm.
    
    PIESNO (Probabilistic Identification and Estimation of Noise) finds
    the background voxels to estimate the noise standard deviation (sigma).
    The SNR is calculated as the mean signal of the foreground (brain) voxels
    divided by the estimated sigma.
    
    Args:
        dwi_tensor: Input DWI tensor of shape (B, C, H, W) or (C, H, W).
        N: Number of receiver coils (default 1 for magnitude/Rician noise).
           
    Returns:
        torch.Tensor: A tensor of shape (B,) containing the estimated SNR for each batch item,
                      or a scalar if no batch dimension.
    """
    is_batched = dwi_tensor.dim() == 4
    if not is_batched:
        dwi_tensor = dwi_tensor.unsqueeze(0)
        
    B, C, H, W = dwi_tensor.shape
    device = dwi_tensor.device
    snr_estimates = []
    
    # Process each item in the batch
    for b in range(B):
        # dipy piesno expects numpy array, typical shape (X, Y, Z, D) or similar.
        # We'll pass (H, W, C)
        data_np = dwi_tensor[b].detach().cpu().numpy()
        data_np = np.transpose(data_np, (1, 2, 0))  # (H, W, C)
        
        try:
            # Run PIESNO to get sigma and the background mask
            sigma, mask = piesno(data_np, N=N, return_mask=True)
            
            # If PIESNO fails to find background or returns 0
            if sigma <= 1e-6:
                snr_estimates.append(100.0) # Assume near-clean if noise is imperceptible
                continue
                
            # Foreground (brain) is the inverse of the background mask
            foreground_mask = ~mask
            
            if not np.any(foreground_mask):
                # Fallback if masking fails completely
                mean_signal = np.mean(data_np)
            else:
                mean_signal = np.mean(data_np[foreground_mask])
                
            snr_est = mean_signal / sigma
            
            # Clip to our curriculum bounds [0.1, 100.0]
            snr_est = np.clip(snr_est, 0.1, 100.0)
            snr_estimates.append(float(snr_est))
            
        except Exception as e:
            # Fallback if PIESNO throws an error (e.g. data is fully zero)
            snr_estimates.append(100.0)
            
    snr_tensor = torch.tensor(snr_estimates, dtype=torch.float32, device=device)
    
    if not is_batched:
        return snr_tensor[0]
        
    return snr_tensor
