#!/usr/bin/env python3
"""
train_warmup.py

Supervised warm-up training for the IterativeRLModel (NO RL).

This trains the full iterative model with:
  ✅ Train/Validation split (80/20 = 14 train / 4 val subjects)
  ✅ Noise curriculum (5% → 10% → 15%)
  ✅ Supervised loss only (L_FW + L_ICVF + L_Sobel + L_aux)
  ✅ No RL agent — model uses default denoising strength
  ✅ CosineAnnealingLR scheduler
  ✅ Gradient clipping for PDE stability
  ✅ Validation-driven checkpointing (best, last, periodic)
  ✅ Epoch-wise metrics logging

The resulting checkpoint is loaded by train_rl_model.py for RL fine-tuning.

Usage:
    python train_warmup.py
"""

import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import random
import yaml
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import RLSliceWiseDWIData
from data.noise_curriculum import NoiseCurriculum
from models.iterative_model import IterativeRLModel
from losses.losses import IterativeRefinementLoss
from validate import run_validation
from utils.checkpointing import (save_checkpoint, save_periodic_checkpoint,
                                  save_metrics_log)


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
def load_config(config_path: str = None) -> dict:
    """Load YAML config, falling back to default_config.yaml."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "config", "default_config.yaml"
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────
# Subject Discovery
# ──────────────────────────────────────────────────────────────────────
def discover_subjects(cfg: dict) -> list:
    """Scan data directories and build the subject list.

    Matches the RDT-Net baseline subject discovery pattern.
    Uses HPC paths first, local fallback.
    """
    paths = cfg["paths"]

    # Try HPC paths first
    base_path = paths["hpc"]["signal_root"]
    gt_fw_root = paths["hpc"]["gt_fw_root"]
    gt_icvf_root = paths["hpc"]["gt_icvf_root"]
    fc_pred_root = paths["hpc"]["fc_pred_root"]

    is_local = False
    if not os.path.isdir(base_path):
        # Fallback to local
        base_path = os.path.join(paths["local"]["dataset_root"], "old_subjects", "training")
        gt_fw_root = os.path.join(paths["local"]["dataset_root"],
                                  "old_subjects", "FWE")
        gt_icvf_root = os.path.join(paths["local"]["dataset_root"],
                                    "old_subjects", "SMT_GT_train")
        fc_pred_root = os.path.join(paths["local"]["dataset_root"],
                                    "old_subjects")
        is_local = True
        print(f"⚠️  HPC paths not found, using local: {base_path}")

    subjects = []
    sampled_log = {}

    for sub in sorted(os.listdir(base_path)):
        subdir = os.path.join(base_path, sub)
        if not os.path.isdir(subdir) or sub.startswith("."):
            continue

        if is_local:
            diff_path = subdir
            signal_path = os.path.join(diff_path, "dwi.nii.gz")
            mask_path = os.path.join(diff_path, "nodif_brain_mask.nii.gz")
            tissue_seg_path = os.path.join(diff_path, "fast_seg.nii.gz")
        else:
            diff_path = os.path.join(subdir, "T1w", "Diffusion")
            if not os.path.isdir(diff_path):
                continue
            signal_path = os.path.join(diff_path, "dwi_b0_1000.nii.gz")
            mask_path = os.path.join(diff_path, "nodif_brain_mask.nii.gz")
            tissue_seg_path = os.path.join(diff_path, "fast_seg.nii.gz")

        gt_fw_path = os.path.join(gt_fw_root, sub, "free_water_volume.nii.gz")
        gt_icvf_path = os.path.join(gt_icvf_root, sub, "ficvf.nii.gz")

        # Check existence
        required = [signal_path, mask_path, gt_fw_path, gt_icvf_path]
        if not all(os.path.exists(p) for p in required):
            missing = [p for p in required if not os.path.exists(p)]
            print(f"[SKIP] {sub}: missing {[os.path.basename(p) for p in missing]}")
            continue

        # Load to check volume count
        img = nib.load(signal_path)
        if len(img.shape) < 4:
            continue
        total_vols = img.shape[3]
        if total_vols < 91:
            continue

        # Sample 90 non-b0 directions + b0
        all_diff_idx = list(range(1, total_vols))
        sampled_diff_idx = sorted(random.sample(all_diff_idx, 90))
        sampled_idx = [0] + sampled_diff_idx

        subj = {
            "id": sub,
            "signal": signal_path,
            "mask": mask_path,
            "gt_fw": gt_fw_path,
            "gt_icvf": gt_icvf_path,
            "sampled_idx": sampled_idx,
        }

        # Tissue segmentation (optional)
        if os.path.exists(tissue_seg_path):
            subj["tissue_seg"] = tissue_seg_path

        subjects.append(subj)
        sampled_log[sub] = sampled_idx

    # Save sampling log
    log_path = os.path.join(
        os.path.dirname(__file__), "sampled_indices_warmup.json"
    )
    with open(log_path, "w") as f:
        json.dump(sampled_log, f, indent=2)

    return subjects


# ──────────────────────────────────────────────────────────────────────
# Train/Validation Split (deterministic, no sklearn dependency)
# ──────────────────────────────────────────────────────────────────────
def train_val_split(subjects: list, test_size: float = 0.2,
                    random_state: int = 42) -> tuple:
    """Split subjects into train and validation sets.

    Deterministic shuffle + split to match sklearn train_test_split
    behavior without requiring the dependency.

    Args:
        subjects: list of subject dicts.
        test_size: fraction for validation (default 0.2 = 4 subjects).
        random_state: random seed for reproducibility.

    Returns:
        (train_subjects, val_subjects)
    """
    n = len(subjects)
    n_val = max(1, round(n * test_size))
    n_train = n - n_val

    # Deterministic shuffle
    rng = random.Random(random_state)
    indices = list(range(n))
    rng.shuffle(indices)

    train_indices = sorted(indices[:n_train])
    val_indices = sorted(indices[n_train:])

    train_subjects = [subjects[i] for i in train_indices]
    val_subjects = [subjects[i] for i in val_indices]

    return train_subjects, val_subjects


# ──────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────
def train_warmup(cfg: dict, subjects: list):
    """Supervised warm-up training (no RL) with train/val split."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    train_cfg = cfg["training"]["warmup"]
    data_cfg = cfg["data"]
    noise_cfg = cfg["noise"]
    iter_cfg = cfg["iterative"]
    loss_cfg = cfg["loss"]
    ckpt_cfg = cfg["training"].get("checkpointing", {})
    val_cfg = cfg["training"].get("validation", {})

    # ── Train/Validation Split ──
    split_seed = data_cfg.get("split_seed", 42)
    val_frac = data_cfg.get("val_split", 0.2)

    train_subjects, val_subjects = train_val_split(
        subjects, test_size=val_frac, random_state=split_seed
    )
    print(f"\n📊 Train/Val Split:")
    print(f"  Training:   {len(train_subjects)} subjects "
          f"({[s['id'] for s in train_subjects]})")
    print(f"  Validation: {len(val_subjects)} subjects "
          f"({[s['id'] for s in val_subjects]})")

    # ── Noise Curriculum ──
    curriculum = NoiseCurriculum(
        epoch_ranges=noise_cfg["curriculum"]["epoch_ranges"],
        noise_levels=noise_cfg["curriculum"]["noise_levels"],
        adaptive_after_epoch=noise_cfg["curriculum"]["adaptive_after_epoch"],
    )
    print(curriculum)

    # ── Datasets (separate train/val) ──
    train_dataset = RLSliceWiseDWIData(
        subject_list=train_subjects,
        target_H=data_cfg["target_H"],
        target_W=data_cfg["target_W"],
        target_snr=100.0,   # will be set per epoch by curriculum
        noise_types=noise_cfg["types"],
    )
    val_dataset = RLSliceWiseDWIData(
        subject_list=val_subjects,
        target_H=data_cfg["target_H"],
        target_W=data_cfg["target_W"],
        target_snr=100.0,   # validation always clean
        noise_types=noise_cfg["types"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    # ── Model ──
    model = IterativeRLModel(
        K_outer=iter_cfg["K_outer"],
        K_inner=cfg["pde"]["K_inner"],
        base_ch=cfg["vit"]["embed_dim"],
        num_dwi=data_cfg["num_dwi_volumes"],
        vit_depth=cfg["vit"]["depth"],
        vit_heads=cfg["vit"]["num_heads"],
        vit_patch_size=cfg["vit"]["patch_size"],
    ).to(device)

    # Load pretrained components
    ann_path = cfg["paths"]["hpc"].get("pretrained_ann")
    vit_path = cfg["paths"]["hpc"].get("pretrained_vit_rxn")
    if ann_path and os.path.exists(ann_path):
        model.load_pretrained_components(ann_path=ann_path, device=device)
    if vit_path and os.path.exists(vit_path):
        model.load_pretrained_components(vit_path=vit_path, device=device)

    # ── PDE weight initialization for numerical stability ──
    pde_gain = cfg["pde"].get("init_gain", 0.1)
    for name, param in model.named_parameters():
        if 'pde' in name and 'weight' in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param, gain=pde_gain)
    print(f"🔧 PDE weights initialized with Xavier gain={pde_gain}")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {param_count:,}")

    # ── Loss ──
    criterion = IterativeRefinementLoss(
        lambda_fw=loss_cfg["lambda_fw"],
        lambda_icvf=loss_cfg["lambda_icvf"],
        lambda_sobel=loss_cfg["lambda_sobel"],
        lambda_rl=0.0,              # NO RL during warm-up
        lambda_aux=loss_cfg["lambda_aux"],
    )

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["epochs"],
        eta_min=train_cfg["eta_min"],
    )

    # ── Output directory (use HPC if accessible, else local) ──
    hpc_out = cfg["paths"]["hpc"].get("output_dir", "")
    local_out = cfg["paths"]["local"]["output_dir"]
    if hpc_out and os.path.isdir(os.path.dirname(hpc_out)):
        output_dir = hpc_out
    else:
        output_dir = local_out
    os.makedirs(output_dir, exist_ok=True)

    # Backward-compatible model path
    model_path = os.path.join(output_dir, "warmup_best.pth")

    # ── Training state ──
    best_val_mae = float("inf")
    start_epoch = 1
    metrics_history = []
    save_every = ckpt_cfg.get("save_every_n_epochs", 5)

    # Initialize lazy parameters (e.g. ViT pos_emb) via a dummy forward pass
    # so that load_state_dict can match all keys from a saved checkpoint.
    with torch.no_grad():
        dummy_dwi = torch.zeros(1, data_cfg["num_dwi_volumes"],
                                data_cfg["target_H"], data_cfg["target_W"],
                                device=device)
        dummy_mask = torch.zeros(1, 3, data_cfg["target_H"],
                                 data_cfg["target_W"], device=device)
        model(dummy_dwi, dummy_mask)
        del dummy_dwi, dummy_mask
        torch.cuda.empty_cache()

    # Auto-resume from checkpoint if it exists
    best_model_path = os.path.join(output_dir, "warmup_best_model.pth")
    last_model_path = os.path.join(output_dir, "warmup_last_model.pth")

    # Try last model first (most recent), then best, then legacy path
    resume_path = None
    for candidate in [last_model_path, best_model_path, model_path]:
        if os.path.exists(candidate):
            resume_path = candidate
            break

    if resume_path:
        print(f"🔄 Found existing checkpoint at {resume_path}. Resuming...")
        try:
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_mae = ckpt.get("best_mae", float("inf"))
            print(f"   Resuming from Epoch {start_epoch} with Best MAE {best_val_mae:.6f}")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}. Starting from scratch.")

    print(f"\n{'='*70}")
    print(f"WARM-UP TRAINING (Supervised, No RL)")
    print(f"  Epochs: {train_cfg['epochs']}")
    print(f"  K_outer: {iter_cfg['K_outer']}")
    print(f"  K_inner (PDE): {cfg['pde']['K_inner']}")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Train subjects: {len(train_subjects)}")
    print(f"  Val subjects: {len(val_subjects)}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    if start_epoch > train_cfg["epochs"]:
        print("✅ Warm-up training is already complete!")
        return

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        model.train()

        # Update noise level from curriculum
        noise_level = curriculum.get_noise_level(epoch)
        if noise_level is None:
            noise_level = 0.15   # cap at 15% during warm-up
        train_dataset.set_target_snr(noise_level if noise_level > 0 else 100.0)

        epoch_losses = []
        epoch_fw_mae = []
        epoch_icvf_mae = []

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{train_cfg['epochs']} "
                         f"(noise={noise_level*100:.0f}%)")

        for batch in pbar:
            (dwi_noisy, dwi_clean, gt_fw, gt_icvf,
             mask, tissue_masks, fc_prior) = [b.to(device) for b in batch]

            optimizer.zero_grad(set_to_none=True)

            # Forward pass (no RL actions during warm-up)
            output = model(dwi_noisy, tissue_masks, rl_actions=None)

            # Clamp predictions
            output["fw_final"] = torch.clamp(output["fw_final"], 0.0, 1.0)
            output["icvf_final"] = torch.clamp(output["icvf_final"], 0.0, 1.0)

            # Loss (no reward during warm-up)
            total_loss, loss_dict = criterion(output, gt_fw, gt_icvf, mask)

            # NaN guard
            if not torch.isfinite(total_loss):
                print("⚠️  NaN detected — skipping batch")
                optimizer.zero_grad()
                continue

            total_loss.backward()

            # Gradient clipping (critical for PDE stability)
            if train_cfg["use_grad_clip"]:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg["max_grad_norm"]
                )

            optimizer.step()

            # Free GPU memory (critical for shared GPU)
            del output, total_loss
            torch.cuda.empty_cache()

            epoch_losses.append(loss_dict["total"])
            epoch_fw_mae.append(loss_dict["fw_mae"])
            epoch_icvf_mae.append(loss_dict["icvf_mae"])

            pbar.set_postfix({
                "fw": f"{loss_dict['fw_mae']:.5f}",
                "icvf": f"{loss_dict['icvf_mae']:.5f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        scheduler.step()

        # ── Train epoch summary ──
        avg_fw = float(np.mean(epoch_fw_mae))
        avg_icvf = float(np.mean(epoch_icvf_mae))
        avg_loss = float(np.mean(epoch_losses))
        train_combined = avg_fw + avg_icvf

        print(f"  [Train] Epoch {epoch}: loss={avg_loss:.6f} "
              f"FW_MAE={avg_fw:.6f} ICVF_MAE={avg_icvf:.6f}")

        # ── Validation ──
        print(f"  [Val] Running validation on {len(val_subjects)} subjects...")
        val_metrics = run_validation(
            model, val_loader, device, epoch, output_dir,
            save_predictions=val_cfg.get("save_predictions", True),
            save_visualizations=val_cfg.get("save_visualizations", True),
        )

        print(f"  [Val] FW_MAE={val_metrics['fw_mae']:.6f} "
              f"ICVF_MAE={val_metrics['icvf_mae']:.6f} "
              f"Combined={val_metrics['combined_mae']:.6f} "
              f"SSIM_FW={val_metrics['ssim_fw']:.4f} "
              f"SSIM_ICVF={val_metrics['ssim_icvf']:.4f}")

        # ── Metrics logging ──
        epoch_metrics = {
            "epoch": epoch,
            "noise_level": noise_level,
            "train_loss": avg_loss,
            "train_fw_mae": avg_fw,
            "train_icvf_mae": avg_icvf,
            "train_combined_mae": train_combined,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "epoch"},
        }
        metrics_history.append(epoch_metrics)
        save_metrics_log(metrics_history, output_dir, prefix="warmup_")

        # ── Checkpointing ──
        val_combined = val_metrics["combined_mae"]
        is_best = val_combined < best_val_mae

        if is_best:
            best_val_mae = val_combined
            print(f"  ✅ New best validation MAE: {best_val_mae:.6f}")

        # Save checkpoint (best + last)
        save_checkpoint(
            model, optimizer, epoch, val_metrics, output_dir,
            is_best=is_best, scheduler=scheduler, config=cfg,
            prefix="warmup_",
        )

        # Also save to legacy path for backward compatibility
        if is_best:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_mae": best_val_mae,
                "config": cfg,
            }, model_path)

        # Periodic checkpoint
        save_periodic_checkpoint(
            model, optimizer, epoch, val_metrics, output_dir,
            save_every=save_every, scheduler=scheduler, config=cfg,
            prefix="warmup_",
        )

    print(f"\n{'='*70}")
    print(f"Warm-up training complete!")
    print(f"Best validation combined MAE: {best_val_mae:.6f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"{'='*70}\n")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    # Seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    cfg = load_config()
    subjects = discover_subjects(cfg)

    if not subjects:
        print("[ERROR] No subjects found for training.")
        return

    print(f"\n✅ Found {len(subjects)} subjects for warm-up training.")
    train_warmup(cfg, subjects)


if __name__ == "__main__":
    main()
