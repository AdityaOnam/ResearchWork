import os
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import RLSliceWiseDWIData
from models.iterative_model import IterativeRLModel
from train_rl_model import discover_subjects, train_val_split

def run_tests():
    config_path = "config/default_config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = discover_subjects(cfg)
    _, val_subjects = train_val_split(subjects, test_size=0.2, random_state=42)
    
    model = IterativeRLModel(
        K_outer=cfg["iterative"]["K_outer"],
        K_inner=cfg["pde"]["K_inner"],
        base_ch=cfg["vit"]["embed_dim"],
        num_dwi=cfg["data"]["num_dwi_volumes"],
        vit_depth=cfg["vit"]["depth"],
        vit_heads=cfg["vit"]["num_heads"],
        vit_patch_size=cfg["vit"]["patch_size"],
    ).to(device)

    model_path = "checkpoints/rl_model_best.pth"
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    snr_levels = [100.0, 50.0, 20.0, 10.0, 5.0, 1.0, 0.1]
    
    dataset = RLSliceWiseDWIData(
        subject_list=val_subjects,
        target_H=cfg["data"]["target_H"],
        target_W=cfg["data"]["target_W"],
        target_snr=100.0,
        noise_types=cfg["noise"]["types"]
    )
    
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    print("\n--- Predicted K per SNR ---")
    for snr in snr_levels:
        dataset.set_target_snr(snr)
        
        all_predicted_k = []
        all_raw_k = []
        
        for i, batch in enumerate(loader):
            if i >= 10: break # Use 160 slices to get a fast average
            
            dwi_noisy, dwi_clean, gt_fw, gt_icvf, mask, tissue_masks = [b.to(device) for b in batch]
            batch_snr = torch.full((dwi_noisy.shape[0], 1), snr, device=device)
            
            with torch.no_grad():
                output = model(dwi_noisy, tissue_masks, snr=batch_snr)
                
            k_pred_raw = output.get('k_pred_raw')
            if k_pred_raw is not None:
                # Clamp between 1 and K_outer=8
                k_clamped = torch.clamp(torch.round(k_pred_raw), min=1, max=8)
                limits = k_clamped.long().cpu().tolist()
                
                all_raw_k.extend(k_pred_raw.cpu().numpy().tolist())
                all_predicted_k.extend(limits)
        
        k_array = np.array(all_predicted_k)
        raw_array = np.array(all_raw_k)
        mean_k = np.mean(k_array)
        mean_raw = np.mean(raw_array)
        std_k = np.std(k_array)
        unique, counts = np.unique(k_array, return_counts=True)
        dist = dict(zip(unique, counts))
        
        print(f"SNR {snr:5.1f} | Mean Integer K: {mean_k:.2f} ± {std_k:.2f} | Raw Avg: {mean_raw:.2f} | Distribution: {dist}")

if __name__ == "__main__":
    run_tests()
