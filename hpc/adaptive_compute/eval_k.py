import os
import torch
import yaml
from torch.utils.data import DataLoader
from data.dataset import RLSliceWiseDWIData
from models.iterative_model import IterativeRLModel
from train_rl_model import discover_subjects, train_val_split
import numpy as np

def load_config(config_path="config/default_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
subjects = discover_subjects(cfg)
train_subjects, val_subjects = train_val_split(subjects, test_size=0.2, random_state=42)

val_dataset = RLSliceWiseDWIData(
    subject_list=val_subjects,
    target_H=cfg["data"]["target_H"],
    target_W=cfg["data"]["target_W"],
    target_snr=0.1,
    noise_types=cfg["noise"]["types"],
)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = IterativeRLModel(
    K_outer=cfg["iterative"]["K_outer"],
    K_inner=cfg["pde"]["K_inner"],
    base_ch=cfg["vit"]["embed_dim"],
    num_dwi=cfg["data"]["num_dwi_volumes"],
    vit_depth=cfg["vit"]["depth"],
    vit_heads=cfg["vit"]["num_heads"],
    vit_patch_size=cfg["vit"]["patch_size"],
).to(device)

ckpt = torch.load("checkpoints/rl_checkpoint_epoch_5.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

all_k_stars = []
target_snr = 0.1

with torch.no_grad():
    for batch in val_loader:
        (dwi_noisy, dwi_clean, gt_fw, gt_icvf, mask, tissue_masks) = [b.to(device) for b in batch]
        batch_snr = torch.full((dwi_noisy.shape[0], 1), target_snr, device=device)
        output = model(dwi_noisy, tissue_masks, snr=batch_snr, force_k_max=True)
        
        fw_inters = output['fw_intermediates']
        icvf_inters = output['icvf_intermediates']
        B = dwi_noisy.shape[0]
        best_k_idx = torch.zeros(B, dtype=torch.long, device=device)
        min_maes = torch.full((B,), float('inf'), device=device)
        
        for k_idx, (fw_k, icvf_k) in enumerate(zip(fw_inters, icvf_inters)):
            mae_fw = torch.abs(fw_k - gt_fw).mean(dim=(1,2,3))
            mae_icvf = torch.abs(icvf_k - gt_icvf).mean(dim=(1,2,3))
            mae = mae_fw + mae_icvf
            
            update_mask = mae < min_maes
            min_maes[update_mask] = mae[update_mask]
            best_k_idx[update_mask] = k_idx
            
        k_star = (best_k_idx + 1).float()
        all_k_stars.extend(k_star.cpu().numpy().tolist())

k_array = np.array(all_k_stars)
print(f"SNR 0.1 - Total slices evaluated: {len(k_array)}")
print(f"SNR 0.1 - Mean Optimal K: {np.mean(k_array):.2f}")
print(f"SNR 0.1 - Median Optimal K: {np.median(k_array):.2f}")
for i in range(1, 9):
    count = np.sum(k_array == i)
    print(f"K={i}: {count} slices ({count/len(k_array)*100:.1f}%)")
