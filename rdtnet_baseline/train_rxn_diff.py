#!/usr/bin/env python3
"""
train_rxn_diff.py — the RDT-Net free-water baseline (ViT encoder + reaction-diffusion PDE).

This is the trainer behind the original RDT-Net checkpoint `VIT_pde_K5_Rxn_diff.pth`:
free-water ground truth in, free-water prior in (channel 92), free-water fraction out.
The QSSM arms (qssm/train/train_mamba.py) keep its decoder and replace its encoder.

Paths come from the same environment variables as the QSSM code (qssm/paths.py):
    $QSSM_TRAIN_SIGNAL/<sid>/T1w/Diffusion/dwi_b0_{1000,2000,3000}.nii.gz, nodif_brain_mask.nii.gz
    $QSSM_TRAIN_GT/<sid>/free_water_volume.nii.gz
    $QSSM_PRIOR_ROOT/<sid>/predicted_fw_volume_fraction.nii.gz
Checkpoint: $QSSM_CKPT_DIR/VIT_pde_K5_Rxn_diff.pth

    python train_rxn_diff.py

Training script for ViT + DSConv + PDE model with:
 - fixed b0 at index 0
 - random (but reproducible) selection of 90 diffusion volumes per subject
 - 1 FC (free-water prediction) channel appended -> total in_ch = 1(b0)+90(DWIs)+1(fc) = 92

Drop-in replacements / additions were applied to:
 - subject collection (main): produce 'sampled_idx' and per-subject 'bvals'
 - dataset loading: subset volumes according to sampled_idx and preserve fc channel
 - safety checks to ensure consistent channel counts
"""
import os
import random
import json
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qssm"))
import paths  # noqa: E402  -- env-var paths; also puts qssm/models (shared PDE decoder) on sys.path

from pde_block import get_default_pde_args  # noqa: E402
from global_layer_rxn_diff import GlobalFeatureBlock_Diffusion  # noqa: E402
from building_blocks import *  # noqa: E402,F403

# ------------------ Device & seeds ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
print(f"Using device: {device}")

# ------------------ Dataset ------------------
class SliceWiseDWIData(Dataset):
    def __init__(self, subject_list, target_H=174, target_W=145, eps_b0=1e-10, verbose=False):
        """
        subject_list: list of dicts with keys:
           'signal'   : path to DWI nifti
           'mask'     : path to mask nifti
           'gt'       : path to GT free-water volume nifti
           'fc_pred'  : path to predicted fc volume nifti
           'sampled_idx': list of volume indices to keep from signal (len == 91)
           optionally: 'bvals' : per-subject bvals list (len == 91)
        """
        self.data = []
        self.target_H = target_H
        self.target_W = target_W

        for subject in tqdm(subject_list, desc="Loading Subjects"):
            sid = subject.get('id', 'N/A')
            try:
                signal_path = subject['signal']
                gt_path = subject['gt']
                mask_path = subject['mask']
                fc_path = subject['fc_pred']
                sampled_idx = subject.get('sampled_idx', None)
                if sampled_idx is None:
                    raise ValueError("subject missing 'sampled_idx'")

                # load volumes
                img = nib.load(signal_path)
                signal_full = img.get_fdata()  # (X, Y, Z, V)
                if signal_full.ndim != 4:
                    raise ValueError(f"Signal nifti not 4D for subject {sid}: shape {signal_full.shape}")

                if max(sampled_idx) >= signal_full.shape[3]:
                    raise ValueError(f"sampled_idx contains index >= available volumes for {sid}")

                # subset volumes -> now should be shape (X,Y,Z,91)
                signal = signal_full[..., sampled_idx].astype(np.float64)
                dims = signal.shape
                nvox = int(np.prod(dims[:3]))
                nvols = dims[3]  # expect 91

                # sanitize numeric issues and b0 normalization
                sig2d = np.reshape(signal, (nvox, nvols)).astype(np.float64)
                sig2d[np.isneginf(sig2d)] = 0.0
                sig2d[np.isposinf(sig2d)] = 0.0
                sig2d[np.isnan(sig2d)] = 0.0
                sig2d[sig2d < 0.0] = 0.0

                b0 = sig2d[:, 0:1].astype(np.float64)
                tiny_mask = (np.abs(b0).reshape(-1) < eps_b0)
                if np.any(tiny_mask):
                    b0[tiny_mask, 0] = eps_b0

                sig2d = sig2d / b0
                sig2d = np.clip(sig2d, 0.0, 1.0)
                norm4d = np.reshape(sig2d.astype(np.float32), dims)

                # reorder to (S, C, H, W)
                signal_slices = np.transpose(norm4d, (0, 3, 1, 2)).astype(np.float32)  # (S, C, H, W)

                # load FC, GT, mask and transpose slices to match (S,H,W)
                fc_img = nib.load(fc_path)
                fc_data = fc_img.get_fdata()
                # if fc has trailing singleton volume dimension, squeeze it
                if fc_data.ndim == 4 and fc_data.shape[-1] == 1:
                    fc_data = np.squeeze(fc_data, axis=-1)
                if fc_data.ndim != 3:
                    raise ValueError(f"FC prediction must be 3D (S,H,W) after squeeze for {sid}, got {fc_data.shape}")
                fc_slices = np.transpose(fc_data, (0, 1, 2)).astype(np.float32)  # (S, H, W)

                gt_img = nib.load(gt_path)
                gt_data = gt_img.get_fdata()
                if gt_data.ndim == 4 and gt_data.shape[-1] == 1:
                    gt_data = np.squeeze(gt_data, axis=-1)
                gt_slices = np.transpose(gt_data, (0, 1, 2)).astype(np.float32)

                mask_img = nib.load(mask_path)
                mask_data = mask_img.get_fdata()
                if mask_data.ndim == 4 and mask_data.shape[-1] == 1:
                    mask_data = np.squeeze(mask_data, axis=-1)
                mask_slices = np.transpose(mask_data, (0, 1, 2)).astype(np.float32)

                # Basic shape checks
                S, C, H, W = signal_slices.shape  # S slices, C volumes (should be 91)
                if C != 91:
                    # allow a recoverable shape but warn
                    raise ValueError(f"Expected 91 volumes after sampling (1 b0 + 90 DWIs) but got C={C} for {sid}")

                if fc_slices.shape[0] != S or gt_slices.shape[0] != S or mask_slices.shape[0] != S:
                    raise ValueError(f"Slice count mismatch for {sid}: signal S={S}, fc S={fc_slices.shape[0]}, gt S={gt_slices.shape[0]}, mask S={mask_slices.shape[0]}")

                # build per-slice examples
                for i in range(S):
                    dwi_slice = signal_slices[i]          # (C, H, W)
                    fc_slice = fc_slices[i][None, ...]    # (1, H, W)
                    full_input = np.concatenate([dwi_slice, fc_slice], axis=0)  # (C+1=92, H, W)

                    # get gt and mask slices
                    gt_slice = gt_slices[i]
                    mask_slice = mask_slices[i]

                    # pad/crop to target H,W
                    current_H, current_W = full_input.shape[1], full_input.shape[2]
                    pad_H = max(0, self.target_H - current_H)
                    pad_W = max(0, self.target_W - current_W)

                    if pad_H > 0 or pad_W > 0:
                        pad_top = pad_H // 2
                        pad_bottom = pad_H - pad_top
                        pad_left = pad_W // 2
                        pad_right = pad_W - pad_left
                        full_input = np.pad(full_input, ((0,0),(pad_top,pad_bottom),(pad_left,pad_right)), mode='constant')
                        gt_slice_p = np.pad(gt_slice, ((pad_top,pad_bottom),(pad_left,pad_right)), mode='constant')
                        mask_slice_p = np.pad(mask_slice, ((pad_top,pad_bottom),(pad_left,pad_right)), mode='constant')
                    else:
                        start_H = (current_H - self.target_H)//2
                        start_W = (current_W - self.target_W)//2
                        full_input = full_input[:, start_H:start_H+self.target_H, start_W:start_W+self.target_W]
                        gt_slice_p = gt_slice[start_H:start_H+self.target_H, start_W:start_W+self.target_W]
                        mask_slice_p = mask_slice[start_H:start_H+self.target_H, start_W:start_W+self.target_W]

                    self.data.append((
                        torch.tensor(full_input, dtype=torch.float32),
                        torch.tensor(gt_slice_p[None, ...], dtype=torch.float32),
                        torch.tensor((mask_slice_p > 0.5)[None, ...].astype(np.float32), dtype=torch.float32)
                    ))

                if verbose:
                    print(f"Loaded subj {sid}: slices={S}, vols={C}, final in_ch={(C+1)}")

            except Exception as e:
                print(f"[ERROR] loading subject {sid}: {e}")

        if len(self.data) == 0:
            raise RuntimeError("No data loaded into dataset. Check subject list and paths.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ------------------ Simple ViT and helpers (same as training) ------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x_att = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + x_att
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleViT(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size=4, depth=4, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        assert patch_size >= 1 and isinstance(patch_size, int)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.depth = depth
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.pos_emb = None
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_p = self.patch_embed(x)
        Hp, Wp = x_p.shape[2], x_p.shape[3]
        N = Hp * Wp
        tokens = x_p.flatten(2).transpose(1, 2)   # (B, N, D)

        if (self.pos_emb is None) or (self.pos_emb.shape[1] != N):
            pe = torch.zeros(1, N, self.embed_dim, device=x.device)
            nn.init.trunc_normal_(pe, std=0.02)
            self.pos_emb = nn.Parameter(pe)

        tokens = tokens + self.pos_emb
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        tokens = tokens.transpose(1, 2).view(B, self.embed_dim, Hp, Wp)
        tokens_up = nn.functional.interpolate(tokens, size=(H, W), mode='bilinear', align_corners=False)
        return tokens_up


class DSConvBlock(nn.Module):
    def __init__(self, channels, expansion=1.0, use_bn=True):
        super().__init__()
        mid = max(1, int(channels * expansion))
        self.use_bn = use_bn
        layers = [
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        ]
        self.net = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        return self.act(out + x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels // reduction), bias=False)
        self.fc2 = nn.Linear(max(1, channels // reduction), channels, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))
        s = self.act(self.fc1(s))
        s = self.sig(self.fc2(s))
        s = s.view(b, c, 1, 1)
        return x * s


class ChannelGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, max(1, channels // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // 2), channels),
            nn.Sigmoid()
        )

    def forward(self, a, b):
        sa = a.mean(dim=(2,3))
        sb = b.mean(dim=(2,3))
        s = torch.cat([sa, sb], dim=1)
        g = self.fc(s).unsqueeze(-1).unsqueeze(-1)
        return g * a + (1.0 - g) * b


class DecoderRefine(nn.Module):
    def __init__(self, base_ch, out_channels=1, use_g1=True, n_refine=2):
        super().__init__()
        self.use_g1 = use_g1
        in_ch = base_ch * (3 if use_g1 else 2)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )
        self.refines = nn.Sequential(*[DSConvBlock(base_ch) for _ in range(n_refine)])
        self.se = SEBlock(base_ch, reduction=8)
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, max(1, base_ch // 2), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, base_ch // 2), out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g2, x0, g1=None):
        if self.use_g1 and g1 is None:
            raise ValueError("DecoderRefine expects g1 if use_g1=True")
        if self.use_g1:
            cat = torch.cat([g2, x0, g1], dim=1)
        else:
            cat = torch.cat([g2, x0], dim=1)
        x = self.reduce(cat)
        x = self.refines(x)
        x = self.se(x)
        out = self.head(x)
        return out


class ConvGlobalSkipModel(nn.Module):
    def __init__(self, in_channels=92, base_ch=64, out_channels=1, K=5,
                 vit_patch_size=4, vit_depth=4, vit_heads=8, aux_out=False):
        super().__init__()
        self.in_vit = SimpleViT(in_ch=in_channels, embed_dim=base_ch,
                                patch_size=vit_patch_size, depth=vit_depth, num_heads=vit_heads)

        self.res1 = DSConvBlock(base_ch)
        pde_args1 = get_default_pde_args(in_chs=base_ch, out_chs=base_ch)
        pde_args1['K'] = K
        pde_args1['constant_Dxy'] = True
        self.global1 = GlobalFeatureBlock_Diffusion(base_ch, pde_args1)

        self.res2 = DSConvBlock(base_ch)
        pde_args2 = get_default_pde_args(in_chs=base_ch, out_chs=base_ch)
        pde_args2['K'] = K
        pde_args2['constant_Dxy'] = True
        self.global2 = GlobalFeatureBlock_Diffusion(base_ch, pde_args2)

        self.gate1 = ChannelGate(base_ch)
        self.gate2 = ChannelGate(base_ch)
        self.se1 = SEBlock(base_ch, reduction=8)
        self.se2 = SEBlock(base_ch, reduction=8)
        self.decoder = DecoderRefine(base_ch=base_ch, out_channels=out_channels, use_g1=True, n_refine=2)

        self.aux_out = aux_out
        if aux_out:
            self.aux_head = nn.Sequential(
                nn.Conv2d(base_ch, base_ch//2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_ch//2, out_channels, kernel_size=1),
                nn.Sigmoid()
            )

        self.alpha = 0.5

    def forward(self, x):
        x0 = self.in_vit(x)
        s0_local = self.res1(x0)
        fused1 = self.gate1(s0_local, x0)
        fused1 = self.se1(fused1)
        try:
            g1 = self.global1(fused1, x0)
        except TypeError:
            g1 = self.global1(fused1)

        s_mid_local = self.res2(g1)
        fused2 = self.gate2(s_mid_local, x0)
        fused2 = self.se2(fused2)
        try:
            g2 = self.global2(fused2, x0)
        except TypeError:
            g2 = self.global2(fused2)

        out = self.decoder(g2, x0, g1)
        if self.aux_out:
            aux = self.aux_head(g1)
            return out, aux
        return out


# ------------------ Losses & physics helper ------------------


def RMSELoss(y_pred, y_true, eps=1e-6):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2) + eps)


def MAELoss_same_logic(y_pred, y_true, eps=1e-12):
    return torch.mean(torch.abs(y_pred - y_true)) + 0.0


def sobel_loss(pred, target):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=pred.device, dtype=pred.dtype).view(1,1,3,3)
    ky = kx.transpose(2,3)
    gx_pred = nn.functional.conv2d(pred, kx, padding=1)
    gy_pred = nn.functional.conv2d(pred, ky, padding=1)
    gx_t = nn.functional.conv2d(target, kx, padding=1)
    gy_t = nn.functional.conv2d(target, ky, padding=1)
    return torch.mean(torch.abs(gx_pred - gx_t) + torch.abs(gy_pred - gy_t))


def physics_reconstruction_loss(S_obs, f_pred, bvals, mask, Dfw=3e-3, eps=1e-6):
    device = S_obs.device
    bvals_arr = torch.tensor(np.asarray(bvals).astype(np.float32), device=device).reshape(1, -1)  # (1,nvols)
    Sfw_row = torch.exp(-Dfw * bvals_arr).reshape(1, -1, 1, 1)  # (1,nvols,1,1)
    f = f_pred.clamp(0.0, 1.0)
    denom = (1.0 - f) + eps
    f_exp = f.expand(-1, S_obs.shape[1], -1, -1)
    Sfw_exp = Sfw_row.expand(S_obs.shape[0], -1, -1, -1)
    St_norm = (S_obs - f_exp * Sfw_exp) / denom
    St_norm = torch.clamp(St_norm, 0.0, 1.0)
    S_recon = f_exp * Sfw_exp + (1.0 - f_exp) * St_norm
    mask_exp = mask.expand(-1, S_obs.shape[1], -1, -1)
    mse = torch.mean(((S_recon - S_obs) * mask_exp) ** 2)
    return mse


# ------------------ Train Loop ------------------

def train_model(subjects, bvals, lr=1e-4, batch_size=2, epochs=200, K=5,
                recon_weight=0.0, sobel_weight=0.02, num_workers=4):

    dataset = SliceWiseDWIData(subjects)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    in_ch = dataset[0][0].shape[0]
    nvols = in_ch - 1
    assert nvols == len(bvals), f"bvals ({len(bvals)}) != nvols ({nvols})"

    model = ConvGlobalSkipModel(in_channels=in_ch, K=K).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_rmse = float("inf")
    model_path = paths.ckpt(f"VIT_pde_K{K}_Rxn_diff.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        batch_rmse, batch_mae = [], []

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} | K={K}")

        for x, y, m in pbar:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)

            optimizer.zero_grad(set_to_none=True)

            out = torch.clamp(model(x), 0.0, 1.0)

            # masked losses
            masked_pred = out * m
            masked_gt   = y * m

            loss_rmse = RMSELoss(masked_pred, masked_gt)
            loss_mae  = MAELoss_same_logic(masked_pred, masked_gt)

            # Sobel loss (safe)
            loss_sobel = sobel_loss(out * m, y * m)
            loss_sobel = torch.nan_to_num(loss_sobel, nan=0.0)

            total_loss = loss_rmse + sobel_weight * loss_sobel

            # NaN / Inf guard
            if not torch.isfinite(total_loss):
                print("⚠️ NaN detected — skipping batch")
                optimizer.zero_grad()
                continue

            total_loss.backward()

            # 🔐 critical for PDE stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            batch_rmse.append(loss_rmse.item())
            batch_mae.append(loss_mae.item())

            pbar.set_postfix(
                rmse=f"{loss_rmse.item():.5f}",
                mae=f"{loss_mae.item():.5f}"
            )

        epoch_rmse = float(np.mean(batch_rmse))
        epoch_mae  = float(np.mean(batch_mae))

        print(f"Epoch {epoch+1}: RMSE={epoch_rmse:.6f} | MAE={epoch_mae:.6f}")

        if epoch_rmse < best_rmse:
            best_rmse = epoch_rmse
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved best model (RMSE={best_rmse:.6f})")


# ------------------ Main ------------------
def main():
    base_path = paths.TRAIN_SIGNAL
    fc_pred_root = paths.PRIOR_ROOT

    subjects = []
    sampled_log = {}

    for sub in sorted(os.listdir(base_path)):
        subdir = os.path.join(base_path, sub)
        if not os.path.isdir(subdir) or sub.startswith('.'):
            continue

        diff_path = os.path.join(subdir, "T1w", "Diffusion")
        if not os.path.isdir(diff_path):
            continue

        # candidate DWI files (choose one DWI file per-subject)
        dwi_options = [os.path.join(diff_path, f"dwi_b0_{val}.nii.gz") for val in [1000, 2000, 3000]]
        dwi_options = [p for p in dwi_options if os.path.exists(p)]
        if not dwi_options:
            continue

        mask_path = os.path.join(diff_path, "nodif_brain_mask.nii.gz")
        gt_path = os.path.join(paths.TRAIN_GT, sub, "free_water_volume.nii.gz")
        fc_pred_path = os.path.join(fc_pred_root, sub, "predicted_fw_volume_fraction.nii.gz")

        if not all(os.path.exists(p) for p in [mask_path, gt_path, fc_pred_path]):
            continue

        # randomly pick one DWI file among available (deterministic due to seed)
        signal_path = random.choice(dwi_options)
        print(f"selected DWI for subj {sub}: {signal_path}")

        # inspect number of volumes
        img = nib.load(signal_path)
        # ensure 4D
        if len(img.shape) < 4:
            print(f"[WARN] {sub} DWI is not 4D. Skipping.")
            continue
        total_vols = img.shape[3]

        if total_vols < 91:
            print(f"[WARN] Subject {sub} DWI has only {total_vols} volumes — need >= 91 (1 b0 + 90 DWIs). Skipping.")
            continue

        # sample indices: keep 0 (b0) fixed and randomly choose 90 from rest
        all_diff_idx = list(range(1, total_vols))
        sampled_diff_idx = sorted(random.sample(all_diff_idx, 90))
        sampled_idx = [0] + sampled_diff_idx  # maintain order (b0 then chosen directions)

        # create a per-subject assumed bvals (0 + 90 x 1000)
        bvals_sub = [0.0] + [1000.0] * (len(sampled_diff_idx))

        subjects.append({
            'id': sub,
            'signal': signal_path,
            'mask': mask_path,
            'gt': gt_path,
            'fc_pred': fc_pred_path,
            'sampled_idx': sampled_idx,
            'bvals': bvals_sub
        })

        sampled_log[sub] = sampled_idx

    if not subjects:
        print("[ERROR] No subjects found for training.")
        return

    # optional: write sampling log (for reproducibility)
    log_path = "sampled_indices_per_subject.json"
    with open(log_path, 'w') as fh:
        json.dump(sampled_log, fh, indent=2)
    print(f"Saved sampled indices to {log_path}")

    # For convenience require all subjects to have same sampled length (they should)
    lengths = [len(s['sampled_idx']) for s in subjects]
    if len(set(lengths)) != 1:
        raise RuntimeError(f"Unequal sampled_idx lengths across subjects: {set(lengths)}")
    assert lengths[0] == 91, f"Expected 91 volumes per subject after sampling, got {lengths[0]}"

    # global bvals (since all subjects have same chosen length)
    bvals = [0.0] + [1000.0] * 90

    print(f"✅ Found {len(subjects)} subjects. Training will use in_ch=92 (91 vols + 1 fc).")

    for K in [5]:
        print(f"\n🚀 Training with K={K}")
        train_model(subjects, bvals, K=K)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""def train_model(subjects, bvals, lr=1e-4, batch_size=2, epochs=200, K=5,
                recon_weight=0, sobel_weight=0.02, num_workers=4):             # give recon_weight=0 , previously 0.5
    dataset = SliceWiseDWIData(subjects)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    in_ch = dataset[0][0].shape[0]
    nvols = in_ch - 1
    assert nvols == len(bvals), f"Global bvals length ({len(bvals)}) must match nvols ({nvols})"

    model = ConvGlobalSkipModel(in_channels=in_ch, K=K).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_rmse = float("inf")
    model_path = f"VIT_pde_K{K}_Rxn_diff.pth"  # without recon loss in name

    for epoch in range(epochs):
        model.train()
        batch_rmse, batch_mae = [], []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} | K={K}")
        for x, y, m in pbar:
            x, y, m = x.to(device), y.to(device), m.to(device)
            out = torch.clamp(model(x), 0, 1)
            masked_pred = out * m
            masked_gt = y * m

            loss_ = RMSELoss(masked_pred, masked_gt)
            loss_mae = MAELoss_same_logic(masked_pred, masked_gt)

            # S_obs: first nvols channels of x
            S_obs = x[:, :nvols, :, :]

            loss_recon = physics_reconstruction_loss(S_obs, out, bvals, m)
            loss_sobel = sobel_loss(out*m, y*m)
            total_loss = 1.0 * loss_f + recon_weight * loss_recon + sobel_weight * loss_sobel

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_rmse.append(float(loss_f.item()))
            batch_mae.append(float(loss_mae.item()))
            pbar.set_postfix(rmse=f"{loss_f.item():.6f}", mae=f"{loss_mae.item():.6f}", recon=f"{loss_recon.item():.6f}")

        epoch_rmse = float(np.mean(batch_rmse))
        epoch_mae = float(np.mean(batch_mae))
        print(f"Epoch {epoch+1}: RMSE={epoch_rmse:.6f} | MAE={epoch_mae:.6f}")

        if epoch_rmse < best_rmse:
            best_rmse = epoch_rmse
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved best model (RMSE={best_rmse:.6f})") 

        for epoch in range(epochs):
            model.train()
            batch_rmse, batch_mae = [], []

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} | K={K}")
        for x, y, m in pbar:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)

        optimizer.zero_grad(set_to_none=True)

        out = model(x)
        out = torch.clamp(out, 0.0, 1.0)

        # masked losses
        masked_pred = out * m
        masked_gt   = y * m

        loss_rmse = RMSELoss(masked_pred, masked_gt)
        loss_mae  = MAELoss_same_logic(masked_pred, masked_gt)


        # sobel loss (safe)
        loss_sobel = sobel_loss(out * m, y * m)
        loss_sobel = torch.nan_to_num(loss_sobel, nan=0.0)

        total_loss = loss_rmse + sobel_weight * loss_sobel

        # NaN guard
        if not torch.isfinite(total_loss):
            print("⚠️ NaN detected — skipping batch")
            optimizer.zero_grad()
            continue

        total_loss.backward()

        # 🔐 gradient clipping (VERY IMPORTANT for PDE)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_rmse.append(loss_rmse.item())
        batch_mae.append(loss_mae.item())

        pbar.set_postfix(
            rmse=f"{loss_rmse.item():.5f}",
            mae=f"{loss_mae.item():.5f}"
        )

    epoch_rmse = float(np.mean(batch_rmse))
    epoch_mae = float(np.mean(batch_mae))

    print(f"Epoch {epoch+1}: RMSE={epoch_rmse:.6f} | MAE={epoch_mae:.6f}")

    if epoch_rmse < best_rmse:
        best_rmse = epoch_rmse
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved best model (RMSE={best_rmse:.6f})") """


