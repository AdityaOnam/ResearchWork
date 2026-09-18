
#!/usr/bin/env python3
"""
test_cnn_pde_fixed_dirs.py

Test / inference script for CNN + PDE model.
- Fixed b0 at index 0
- Deterministic selection of non-b0 diffusion directions (evenly spaced)
- Zero-padding of unselected directions
- Subsets bvals to match selected channels
- Computes per-slice and per-subject RMSE & MAE
"""
import os
import sys
import csv
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm

from pde_block import get_default_pde_args
from global_layer_rxn_diff import GlobalFeatureBlock_Diffusion
from building_blocks import *


# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
directions_to_test = [ 91]

# ------------------ Simple ViT (same as training) ------------------
class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                                batch_first=True, dropout=dropout)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, int(dim * mlp_ratio)),
            torch.nn.GELU(),
            torch.nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x_att = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + x_att
        x = x + self.mlp(self.norm2(x))
        return x

class SimpleViT(torch.nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size=4, depth=4, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        assert patch_size >= 1 and isinstance(patch_size, int)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = torch.nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.depth = depth
        self.blocks = torch.nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.pos_emb = None
        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_p = self.patch_embed(x)                 # (B, D, Hp, Wp)
        Hp, Wp = x_p.shape[2], x_p.shape[3]
        N = Hp * Wp

        tokens = x_p.flatten(2).transpose(1, 2)   # (B, N, D)

        # Lazy init positional embedding
        if (self.pos_emb is None) or (self.pos_emb.shape[1] != N):
            pe = torch.zeros(1, N, self.embed_dim, device=x.device)
            torch.nn.init.trunc_normal_(pe, std=0.02)
            self.pos_emb = torch.nn.Parameter(pe)

        tokens = tokens + self.pos_emb
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)                # (B, N, D)
        tokens = tokens.transpose(1, 2).view(B, self.embed_dim, Hp, Wp)
        tokens_up = torch.nn.functional.interpolate(tokens, size=(H, W), mode='bilinear', align_corners=False)
        return tokens_up

# ------------------ Lightweight DSConv + helpers (same as training) ------------------
class DSConvBlock(torch.nn.Module):
    def __init__(self, channels, expansion=1.0, use_bn=True):
        super().__init__()
        mid = max(1, int(channels * expansion))
        self.use_bn = use_bn
        layers = []
        layers.append(torch.nn.Conv2d(channels, mid, kernel_size=1, bias=False))
        layers.append(torch.nn.BatchNorm2d(mid) if use_bn else torch.nn.Identity())
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False))
        layers.append(torch.nn.BatchNorm2d(mid) if use_bn else torch.nn.Identity())
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv2d(mid, channels, kernel_size=1, bias=False))
        layers.append(torch.nn.BatchNorm2d(channels) if use_bn else torch.nn.Identity())
        self.net = torch.nn.Sequential(*layers)
        self.act = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        return self.act(out + x)

class SEBlock(torch.nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = torch.nn.Linear(channels, max(1, channels // reduction), bias=False)
        self.fc2 = torch.nn.Linear(max(1, channels // reduction), channels, bias=False)
        self.act = torch.nn.ReLU(inplace=True)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))
        s = self.act(self.fc1(s))
        s = self.sig(self.fc2(s))
        s = s.view(b, c, 1, 1)
        return x * s

class ChannelGate(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channels * 2, max(1, channels // 2)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(max(1, channels // 2), channels),
            torch.nn.Sigmoid()
        )
    def forward(self, a, b):
        sa = a.mean(dim=(2,3))
        sb = b.mean(dim=(2,3))
        s = torch.cat([sa, sb], dim=1)
        g = self.fc(s).unsqueeze(-1).unsqueeze(-1)
        return g * a + (1.0 - g) * b

class DecoderRefine(torch.nn.Module):
    def __init__(self, base_ch, out_channels=1, use_g1=True, n_refine=2):
        super().__init__()
        self.use_g1 = use_g1
        in_ch = base_ch * (3 if use_g1 else 2)
        self.reduce = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, base_ch, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(base_ch),
            torch.nn.ReLU(inplace=True)
        )
        self.refines = torch.nn.Sequential(*[DSConvBlock(base_ch) for _ in range(n_refine)])
        self.se = SEBlock(base_ch, reduction=8)
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(base_ch, max(1, base_ch // 2), kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(max(1, base_ch // 2), out_channels, kernel_size=1),
            torch.nn.Sigmoid()
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

# ------------------ Model (same architecture as training script) ------------------
class ConvGlobalSkipModel(torch.nn.Module):
    # DIFF vs FW_RDT_NET_SYNTH/src/test_rxn_diff.py: the `encoder`/`enc_kw` kwargs below,
    # and the `order` kwarg on forward(). Everything else in this file is verbatim, so a
    # `diff` against the sibling is auditable. `encoder="vit"` reproduces the sibling
    # exactly, which is what the A0 control run depends on.
    def __init__(self, in_channels=92, base_ch=64, out_channels=1, K=5,
                 vit_patch_size=4, vit_depth=4, vit_heads=8, aux_out=False,
                 encoder="vit", enc_kw=None, dt_live=False):
        super().__init__()
        from encoders import build_encoder
        self.in_vit = build_encoder(encoder, in_channels, base_ch,
                                    patch_size=vit_patch_size, depth=vit_depth,
                                    num_heads=vit_heads, **(enc_kw or {}))
        self.res1 = DSConvBlock(base_ch)
        pde_args1 = get_default_pde_args(in_chs=base_ch, out_chs=base_ch)
        pde_args1['K'] = K
        pde_args1['constant_Dxy'] = True
        pde_args1['dt_live'] = dt_live
        self.global1 = GlobalFeatureBlock_Diffusion(base_ch, pde_args1)

        self.res2 = DSConvBlock(base_ch)
        pde_args2 = get_default_pde_args(in_chs=base_ch, out_chs=base_ch)
        pde_args2['K'] = K
        pde_args2['constant_Dxy'] = True
        pde_args2['dt_live'] = dt_live
        self.global2 = GlobalFeatureBlock_Diffusion(base_ch, pde_args2)

        self.gate1 = ChannelGate(base_ch)
        self.gate2 = ChannelGate(base_ch)
        self.se1 = SEBlock(base_ch, reduction=8)
        self.se2 = SEBlock(base_ch, reduction=8)
        self.decoder = DecoderRefine(base_ch=base_ch, out_channels=out_channels, use_g1=True, n_refine=2)

        self.aux_out = aux_out
        if aux_out:
            self.aux_head = torch.nn.Sequential(
                torch.nn.Conv2d(base_ch, base_ch//2, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(base_ch//2, out_channels, kernel_size=1),
                torch.nn.Sigmoid()
            )
        self.alpha = 0.5

    def forward(self, x, order=None):
        x0 = self.in_vit(x, order)  # (B, base_ch, H, W)
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

# ------------------ Metrics & helpers ------------------
def RMSELoss(y_pred, y_true, eps=1e-6):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2) + eps)

def MAELoss_same_logic(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true)) + 0.0

# ------------------ New prepare function: angular selection + zero-padding ------------------
def prepare_dwi_b0_flat_and_slices(dwi_path, num_dirs_to_keep=None, bvals_path=None, eps_b0=1e-10):
    """
    Load DWI or single-volume b0, normalize to b0, select directions angularly-uniform among non-b0 vols,
    and zero-pad remaining channels so output channel count equals original file's channel count.

    If the input file is 3D and a bvals_path is provided (Option B: bvals exclude the b0),
    we will expand the 3D b0 into a 4D volume with 1 + len(bvals_nonb0) volumes, by
    replicating the b0 into non-b0 positions. (This is synthetic / placeholder behavior.)

    Returns:
      signal_slices_zero_padded: (S, full_nvols, H, W) float32
      dims (original/new)
      affine, header
      dwidata_norm_flat_padded: (nvox, full_nvols) float32 (zeros for unselected columns)
      b0_flat: (nvox,) float32
      sampled_idx: list of selected indices in original file (includes 0 for b0)
    """
    img = nib.load(dwi_path)
    data = img.get_fdata().astype(np.float64)

    # If 3D and bvals provided (Option B), expand to 4D
    if data.ndim == 3:
        if bvals_path is None or not os.path.exists(bvals_path):
            raise ValueError(f"DWI is 3D but no bvals file provided/found for: {dwi_path}")
        # load bvals (assume bvals excludes the b0 per Option B)
        bvals_nonb0 = np.loadtxt(bvals_path).astype(float).ravel()
        n_nonb0 = bvals_nonb0.size
        full_nvols = 1 + n_nonb0
        # replicate b0 into all volumes (volume 0 = actual b0)
        data4d = np.zeros((data.shape[0], data.shape[1], data.shape[2], full_nvols), dtype=np.float64)
        data4d[..., 0] = data
        for vi in range(1, full_nvols):
            # replicate b0 into other volumes (synthetic)
            data4d[..., vi] = data
        data = data4d
        # create a simple header/affine keep same
        dims = data.shape
    elif data.ndim == 4:
        dims = data.shape  # (X, Y, Z, V_full)
    else:
        raise ValueError("DWI must be 3D or 4D")

    nvox = int(np.prod(dims[:3]))
    full_nvols = dims[3]

    # sanitize and flatten
    data2d = np.reshape(data, (nvox, full_nvols)).astype(np.float64)
    data2d[np.isneginf(data2d)] = 0.0
    data2d[np.isposinf(data2d)] = 0.0
    data2d[np.isnan(data2d)] = 0.0
    data2d[data2d < 0.0] = 0.0

    # b0 (index 0) fixed and safe
    b0 = data2d[:, 0:1].astype(np.float64)
    tiny_mask = (np.abs(b0).reshape(-1) < eps_b0)
    if np.any(tiny_mask):
        b0 = b0.copy()
        b0[tiny_mask, 0] = eps_b0

    # normalized full dataset (we will subset later)
    data2d_norm = data2d / b0
    data2d_norm[np.isnan(data2d_norm)] = 0.0
    data2d_norm[data2d_norm < 0.0] = 0.0
    data2d_norm[data2d_norm > 1.0] = 1.0

    # number of non-b0 volumes available
    n_nonb0 = full_nvols - 1
    if n_nonb0 <= 0:
        raise ValueError("No non-b0 volumes found in DWI.")

    # If user requests more non-b0 than available, clamp
    if num_dirs_to_keep is None:
        keep_nonb0 = n_nonb0
    else:
        # input num_dirs_to_keep was number of DWIs (commonly), keep it as number of non-b0s desired
        keep_nonb0 = int(min(num_dirs_to_keep, n_nonb0))

    # Build angularly-uniform indices among non-b0 volumes: choose evenly-spaced indices from 1..(V-1)
    if keep_nonb0 <= 0:
        sampled_nonb0 = []
    else:
        # sample positions (1..full_nvols-1) uniformly spaced (deterministic)
        positions = np.linspace(1, full_nvols - 1, num=keep_nonb0, dtype=int)
        # ensure unique and sorted
        sampled_nonb0 = sorted(np.unique(positions).tolist())

    # final sampled indices include b0 at 0
    sampled_idx = [0] + sampled_nonb0

    # Prepare zero-padded flattened normalized data (nvox, full_nvols)
    dwidata_norm_flat_padded = np.zeros((nvox, full_nvols), dtype=np.float32)
    for idx in sampled_idx:
        dwidata_norm_flat_padded[:, idx] = data2d_norm[:, idx]

    # Build sampled 4D stack with zero padding for unselected channels
    sampled_stack = np.zeros((dims[0], dims[1], dims[2], full_nvols), dtype=np.float32)
    for idx in sampled_idx:
        sampled_stack[..., idx] = np.reshape(data2d_norm[:, idx], (dims[0], dims[1], dims[2]))

    # transpose to slice-first convention used elsewhere: (S, C, H, W) where S=dims[0]
    signal_slices_zero_padded = np.transpose(sampled_stack, (0, 3, 1, 2)).astype(np.float32)  # (S, full_nvols, H, W)

    b0_flat = b0.astype(np.float32).reshape(-1)

    return signal_slices_zero_padded, dims, img.affine, img.header, dwidata_norm_flat_padded, b0_flat, sampled_idx

# ------------------ remove + denorm (unchanged) ------------------
def remove_fw_component_and_denorm_b0(dwidata_norm_flat, b0_flat, f_flat, bvals):
    S_norm = dwidata_norm_flat.astype(np.float64)
    f_arr = f_flat.reshape(-1, 1).astype(np.float64)
    bvals = np.asarray(bvals).ravel().astype(np.float64)
    Dfw = 3e-3
    Sfw_row = np.exp(-Dfw * bvals).reshape(1, -1)
    Scsf = f_arr * Sfw_row

    denom = (1.0 - f_arr)
    denom_safe = denom.copy()
    denom_safe[denom_safe == 0] = 1.0

    St_norm = (S_norm - Scsf) / denom_safe
    St_norm[np.isnan(St_norm)] = 0.0
    St_norm[St_norm < 0.0] = 0.0

    b0_col = np.asarray(b0_flat).reshape(-1, 1).astype(np.float64)
    St = St_norm * b0_col
    St[np.isnan(St)] = 0.0
    St[St < 0.0] = 0.0
    return St.astype(np.float32)

# ------------------ Test loop (updated to use new prepare return) ------------------
def test_model(model_path, test_subjects, output_dir, K, device=device, num_dirs_to_keep=None):
    os.makedirs(output_dir, exist_ok=True)
    
    model = ConvGlobalSkipModel(in_channels=1 + num_dirs_to_keep)
    ck = torch.load(model_path, map_location=device)

    # Extract state_dict safely
    if isinstance(ck, dict):
        if 'state_dict' in ck:
            sd = ck['state_dict']
        elif 'model_state_dict' in ck:
            sd = ck['model_state_dict']
        else:
            sd = ck
    else:
        sd = ck

    # Always define new_sd
    new_sd = {}
    for k, v in sd.items():
        if isinstance(k, str):
            new_sd[k.replace('module.', '')] = v
        else:
            new_sd[k] = v

    model.load_state_dict(new_sd, strict=False)
    model = model.to(device)
    model.eval()


    total_rmse = []
    total_mae = []
    subject_metrics = []
    failed = []

    with torch.no_grad():
        for subj in tqdm(test_subjects, desc=f"Testing K={K}"):
            sid = subj.get('id', 'unknown')
            try:
                # prepare normalized slices and flattened data (now returns sampled_idx too)
                bvals_path = subj.get('bvals_file', None)
                (signal_slices, dims, affine, header, dwidata_norm_flat, b0_flat, sampled_idx) = \
                     prepare_dwi_b0_flat_and_slices(subj['signal'], num_dirs_to_keep, bvals_path)
    

                S, C, H, W = signal_slices.shape
                nvols = C  # full number of channels (includes zero-padded ones)

                mask_img = nib.load(subj.get('mask'))
                mask = mask_img.get_fdata()
                gt_img = nib.load(subj.get('gt'))
                gt = gt_img.get_fdata()
                fc_path = subj.get('fc') if 'fc' in subj else subj.get('fc_pred', None)
                if fc_path is None:
                    raise ValueError("No fc/fc_pred entry in subject dict")
                fc_img = nib.load(fc_path)
                fc_vol = fc_img.get_fdata()

                if mask.ndim == 4 and mask.shape[-1] == 1:
                    mask = np.squeeze(mask, axis=-1)
                if gt.ndim == 4 and gt.shape[-1] == 1:
                    gt = np.squeeze(gt, axis=-1)
                if fc_vol.ndim == 4 and fc_vol.shape[-1] == 1:
                    fc_vol = np.squeeze(fc_vol, axis=-1)

                mask_slices = np.transpose(mask, (0,1,2))
                gt_slices = np.transpose(gt, (0,1,2))
                fc_slices = np.transpose(fc_vol, (0,1,2))

                pred_stack = []
                rmse_list = []
                mae_list = []

                # model expected input channels logic (keeps your original behavior)
                model_in_ch = None
                try:
                    model_in_ch = model.in_vit.patch_embed.in_channels
                except Exception:
                    try:
                        model_in_ch = model.decoder.reduce[0].in_channels  # fallback
                    except Exception:
                        model_in_ch = 92
                model_dwi_ch_expected = max(1, model_in_ch - 1)

                for i in range(S):
                    dwi = signal_slices[i]   # shape (full_nvols, H, W) with many zeros if num_dirs_to_keep < full_nvols
                    # reduce/pad dwi channels to model_dwi_ch_expected (this keeps compatibility)
                    ch = dwi.shape[0]
                    H_sig, W_sig = dwi.shape[1], dwi.shape[2]
                    if ch == model_dwi_ch_expected:
                        dwi_for_model = dwi
                    elif ch > model_dwi_ch_expected:
                        dwi_for_model = dwi[:model_dwi_ch_expected, ...]
                    else:
                        pad = np.zeros((model_dwi_ch_expected - ch, H_sig, W_sig), dtype=np.float32)
                        dwi_for_model = np.concatenate([dwi, pad], axis=0)

                    fc_slice = fc_slices[i]
                    fh, fw = fc_slice.shape
                    if fh != H_sig or fw != W_sig:
                        pad_H = max(0, H_sig - fh); pad_W = max(0, W_sig - fw)
                        pt, pb = pad_H//2, pad_H - pad_H//2
                        pl, pr = pad_W//2, pad_W - pad_W//2
                        fc_slice = np.pad(fc_slice, ((pt,pb),(pl,pr)), mode='constant')

                    inp = np.concatenate([dwi_for_model, fc_slice[None, ...]], axis=0)
                    x = torch.from_numpy(inp[None, ...].astype(np.float32)).to(device)

                    y = torch.tensor(gt_slices[i][None, None, ...], dtype=torch.float32).to(device)
                    m = torch.tensor((mask_slices[i] > 0.5)[None, None, ...], dtype=torch.float32).to(device)

                    pred = model(x)
                    if isinstance(pred, (tuple, list)):
                        pred = pred[0]
                    pred = torch.clamp(pred, 0.0, 1.0)

                    rmse_list.append(float(RMSELoss(pred * m, y * m).item()))
                    mae_list.append(float(MAELoss_same_logic(pred * m, y * m).item()))
                    pred_stack.append(pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32))

                fw_vol = np.stack(pred_stack, axis=0)
                fw_vol = np.clip(fw_vol, 0.0, 1.0)
                fw_vol_mask = fw_vol * (mask_slices > 0.5)
                tissue_vol = (1.0 - fw_vol) * (mask_slices > 0.5)

                # bvals handling: subset using sampled_idx (if subject has bvals_file). Otherwise build default full bvals and subset.
                bvals_arr = None
                if 'bvals_file' in subj and subj['bvals_file'] and os.path.exists(subj['bvals_file']):
                    try:
                        bvals_full = np.loadtxt(subj['bvals_file']).astype(float).ravel()
                        # if bvals_full length matches full_nvols, subset.
                        if bvals_full.size == dwidata_norm_flat.shape[1]:
                            bvals_arr = bvals_full[np.array(sampled_idx)].astype(float)
                        elif bvals_full.size == (dwidata_norm_flat.shape[1] - 1):
                            # maybe bvals file excludes b0; handle gracefully by prepending 0
                            bvals_full2 = np.concatenate(([0.0], bvals_full))
                            bvals_arr = bvals_full2[np.array(sampled_idx)].astype(float)
                        else:
                            # fallback: assume all non-b0 are 1000
                            bvals_arr = [0.0] + [1000.0] * (len(sampled_idx)-1)
                    except Exception:
                        bvals_arr = [0.0] + [1000.0] * (len(sampled_idx)-1)
                else:
                    # default full bvals (with same length as full channels) then subset
                    full_bvals_default = np.array([0.0] + [1000.0] * (dwidata_norm_flat.shape[1] - 1), dtype=float)
                    bvals_arr = full_bvals_default[np.array(sampled_idx)].astype(float)

                # prepare f_flat using predicted fw (fw_vol oriented as S x H x W where S corresponds to dims[0])
                f_flat = fw_vol.reshape(-1, 1).astype(np.float32)

                # use dwidata_norm_flat (padded) and subset columns according to sampled_idx for denorm function
                # remove_fw_component_and_denorm_b0 expects dwidata_norm_flat to have same number of columns as bvals passed.
                # We'll build dwidata_norm_flat_sub_by_sampled: (nvox, len(sampled_idx))
                dwidata_norm_flat_sub_by_sampled = dwidata_norm_flat[:, sampled_idx]

                St_flat_sub = remove_fw_component_and_denorm_b0(dwidata_norm_flat_sub_by_sampled, b0_flat, f_flat, bvals_arr)
                # St_flat_sub shape: (nvox, len(sampled_idx)) -> reshape to (X,Y,Z,N_sel)
                N_sel = len(sampled_idx)
                St4d_masked = St_flat_sub.reshape(dims[0], dims[1], dims[2], N_sel).astype(np.float32)

                # apply mask: create mask_vol corresponding to original spatial dims
                mask_vol = np.zeros((dims[0], dims[1], dims[2]), dtype=bool)
                for s in range(mask_slices.shape[0]):
                    mask_vol[s, :, :] = (mask_slices[s] > 0.5)
                mask_flat = mask_vol.reshape(-1).astype(bool)
                if St4d_masked.ndim == 4:
                    nonbrain_idx = np.where(~mask_flat)[0]
                    if nonbrain_idx.size > 0:
                        St_flat_masked = St_flat_sub.copy()
                        St_flat_masked[nonbrain_idx, :] = 0.0
                        St4d_masked = St_flat_masked.reshape(dims[0], dims[1], dims[2], N_sel).astype(np.float32)

                # Save outputs (fw and tissue volumes have shape (S,H,W) -> map back to (X,Y,Z))
                outdir = os.path.join(output_dir, sid)
                os.makedirs(outdir, exist_ok=True)
                # reconstruct fw volume as training did: S corresponds to dims[0]
                fw_vol_full = np.zeros((int(dims[0]), int(dims[1]), int(dims[2])), dtype=np.float32)
                for s in range(fw_vol.shape[0]):
                    fw_vol_full[s, :, :] = fw_vol[s]

                fw_vol_full = np.nan_to_num(fw_vol_full, nan=0.0)
                fw_vol_full = np.clip(fw_vol_full, 0.0, 1.0)
                fw_masked_full = fw_vol_full * mask_vol
                tissue_masked_full = (1.0 - fw_vol_full) * mask_vol

                try:
                    nib.save(nib.Nifti1Image(fw_masked_full.astype(np.float32), affine, header),
                             os.path.join(outdir, f"predicted_free_water_volume_{len(sampled_idx)}.nii.gz"))
                    nib.save(nib.Nifti1Image(tissue_masked_full.astype(np.float32), affine, header),
                             os.path.join(outdir, f"tissue_volume_fraction_{len(sampled_idx)}.nii.gz"))
                except Exception:
                    nib.save(nib.Nifti1Image(fw_masked_full.astype(np.float32), affine),
                             os.path.join(outdir, f"predicted_free_water_volume_{len(sampled_idx)}.nii.gz"))
                    nib.save(nib.Nifti1Image(tissue_masked_full.astype(np.float32), affine),
                             os.path.join(outdir, f"tissue_volume_fraction_{len(sampled_idx)}.nii.gz"))

                # Save subselected denorm DWI (only selected volumes are meaningful)
                try:
                    nib.save(nib.Nifti1Image(St4d_masked.astype(np.float32), affine, header),
                             os.path.join(outdir, f"fwe_dwi_selected_{len(sampled_idx)}.nii.gz"))
                except Exception:
                    nib.save(nib.Nifti1Image(St4d_masked.astype(np.float32), affine),
                             os.path.join(outdir, f"fwe_dwi_selected_{len(sampled_idx)}.nii.gz"))

                subj_rmse = float(np.mean(rmse_list))
                subj_mae = float(np.mean(mae_list))
                total_rmse.append(subj_rmse)
                total_mae.append(subj_mae)
                subject_metrics.append((sid, subj_rmse, subj_mae))

                print(f"[OK] {sid} | RMSE={subj_rmse:.6f} | MAE={subj_mae:.6f} (slices={S})  sampled_idx_len={len(sampled_idx)}")

            except Exception as e:
                print(f"[FAIL] {sid} -> {e}")
                failed.append((sid, str(e)))
                continue

    avg_rmse = float(np.nanmean(total_rmse)) if len(total_rmse) > 0 else float('nan')
    std_rmse = float(np.nanstd(total_rmse)) if len(total_rmse) > 0 else float('nan')
    avg_mae = float(np.nanmean(total_mae)) if len(total_mae) > 0 else float('nan')
    std_mae = float(np.nanstd(total_mae)) if len(total_mae) > 0 else float('nan')

    print(f"\n✅ Avg RMSE for K={K}: {avg_rmse:.6f} ± {std_rmse:.6f}")
    print(f"✅ Avg MAE for K={K}: {avg_mae:.6f} ± {std_mae:.6f}")

    if failed:
        print(f"\n⚠️ Failed Subjects ({len(failed)}):")
        for sid, err in failed:
            print(f"- {sid}: {err}")

    csv_path = os.path.join(output_dir, f"test_metrics_K{K}_{num_dirs_to_keep}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["SubjectID", "RMSE", "MAE_same_logic"])
        for sid, rmse, mae in subject_metrics:
            writer.writerow([sid, f"{rmse:.6f}", f"{mae:.6f}"])
        writer.writerow([])
        writer.writerow([f"\n✅ Avg RMSE for K={K}: {avg_rmse:.6f} ± {std_rmse:.6f}"])
        writer.writerow([f"✅ Avg MAE for K={K}: {avg_mae:.6f} ± {std_mae:.6f}"])

# ------------------ Main ------------------
def main():
    """Legacy direction-subsampling test of the RDT-Net (ViT) baseline.

    Paths come from qssm/paths.py (environment variables); layout expected under
    $QSSM_TEST_ROOT: Signal/<sid>/T1w/Diffusion/, t_data/<sid>/T1w/Diffusion/nodif_brain_mask,
    Analysis/multi_shell/Ground_Truth/<sid>/free_water_volume.nii.gz.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import paths

    test_root = paths.TEST_ROOT
    fc_root = paths.PRIOR_ROOT
    output_root = paths.results("rdt_net_direction_test")

    test_subjects = []
    for sub in paths.subjects("test"):
        signal_file = os.path.join(test_root, "Signal", sub, "T1w", "Diffusion", "dwi_b0_1000.nii.gz")
        mask_file = os.path.join(test_root, "t_data", sub, "T1w", "Diffusion", "nodif_brain_mask.nii.gz")
        gt_file = os.path.join(test_root, "Analysis", "multi_shell", "Ground_Truth", sub, "free_water_volume.nii.gz")
        fc_pred = os.path.join(fc_root, sub, "dwi_b0_1000", "fw_volume_fraction.nii.gz")
        if all(os.path.exists(p) for p in [signal_file, mask_file, gt_file, fc_pred]):
            test_subjects.append({'id': sub, 'signal': signal_file, 'mask': mask_file, 'gt': gt_file, 'fc': fc_pred})

    if not test_subjects:
        print("[ERROR] No test subjects found.")
        return

    for num_dirs_to_keep in directions_to_test:
        print(f"\n===== Testing with {num_dirs_to_keep} diffusion directions (non-b0 chosen angularly) =====")
        for K in [5]:
            model_path = paths.ckpt(f"VIT_pde_K{K}_Rxn_diff.pth")
            if not os.path.exists(model_path):
                print(f"[WARN] Model not found for K={K}: {model_path}. Skipping.")
                continue
            outdir = os.path.join(output_root, f"K{K}_dirs{num_dirs_to_keep}")
            os.makedirs(outdir, exist_ok=True)
            test_model(model_path, test_subjects, outdir, K=K, device=device, num_dirs_to_keep=num_dirs_to_keep)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
