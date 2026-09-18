#!/usr/bin/env python3
"""
train_rl_model.py

Full RL-augmented training for the IterativeRLModel.

Workflow:
    1. Load warm-up checkpoint (from train_warmup.py).
    2. Initialise PPO agent with enhanced rewards.
    3. Alternate: supervised gradient step + RL policy update.
    4. Reverse extreme-noise curriculum (300% → 0%).
    5. Total loss: L_FW + L_ICVF + L_Sobel + L_RL + L_aux.
    6. Train/Val split: 14 train / 4 validation subjects.
    7. Validation-driven checkpointing.

Usage:
    python train_rl_model.py
    python train_rl_model.py --warmup-ckpt /path/to/warmup_best.pth
"""

import os
import sys
import json
import random
import argparse
import yaml
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import RLSliceWiseDWIData
from data.noise_curriculum import RLNoiseCurriculum
from models.iterative_model import IterativeRLModel
from losses.losses import IterativeRefinementLoss
from rl.rl_agent import PPOAgent
from rl.reward import compute_reward
from rl.policy import ActorCritic
from validate import run_validation
from utils.checkpointing import (save_checkpoint, save_periodic_checkpoint,
                                  save_metrics_log)


# ──────────────────────────────────────────────────────────────────────
# Configuration (reused from train_warmup.py)
# ──────────────────────────────────────────────────────────────────────
def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "config", "default_config.yaml"
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def discover_subjects(cfg):
    """Discover subjects — identical to train_warmup.py."""
    paths = cfg["paths"]
    base_path = paths["hpc"]["signal_root"]
    gt_fw_root = paths["hpc"]["gt_fw_root"]
    gt_icvf_root = paths["hpc"]["gt_icvf_root"]

    is_local = False
    if not os.path.isdir(base_path):
        base_path = os.path.join(paths["local"]["dataset_root"], "old_subjects", "training")
        gt_fw_root = os.path.join(paths["local"]["dataset_root"],
                                  "old_subjects", "FWE")
        gt_icvf_root = os.path.join(paths["local"]["dataset_root"],
                                    "old_subjects", "SMT_GT_train")
        is_local = True
        print(f"⚠️  Using local paths: {base_path}")

    subjects = []
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

        required = [signal_path, mask_path, gt_fw_path, gt_icvf_path]
        if not all(os.path.exists(p) for p in required):
            continue

        img = nib.load(signal_path)
        if len(img.shape) < 4 or img.shape[3] < 91:
            continue

        all_diff_idx = list(range(1, img.shape[3]))
        sampled_idx = [0] + sorted(random.sample(all_diff_idx, 90))

        subj = {
            "id": sub, "signal": signal_path, "mask": mask_path,
            "gt_fw": gt_fw_path, "gt_icvf": gt_icvf_path,
            "sampled_idx": sampled_idx,
        }
        if os.path.exists(tissue_seg_path):
            subj["tissue_seg"] = tissue_seg_path

        subjects.append(subj)

    return subjects


def train_val_split(subjects, test_size=0.2, random_state=42):
    """Deterministic train/val split (matches train_warmup.py)."""
    n = len(subjects)
    n_val = max(1, round(n * test_size))
    n_train = n - n_val

    rng = random.Random(random_state)
    indices = list(range(n))
    rng.shuffle(indices)

    train_indices = sorted(indices[:n_train])
    val_indices = sorted(indices[n_train:])

    return ([subjects[i] for i in train_indices],
            [subjects[i] for i in val_indices])


# ──────────────────────────────────────────────────────────────────────
# RL-augmented training loop
# ──────────────────────────────────────────────────────────────────────
def train_rl(cfg, subjects, warmup_ckpt_path=None):
    """Train with alternating supervised + PPO updates."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    rl_cfg = cfg["training"]["rl_phase"]
    data_cfg = cfg["data"]
    noise_cfg = cfg["noise"]
    iter_cfg = cfg["iterative"]
    loss_cfg = cfg["loss"]
    ppo_cfg = cfg["rl"]
    reward_cfg = cfg.get("reward", {})
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

    # ── Reverse Extreme-Noise Curriculum ──
    rl_noise_cfg = noise_cfg.get("rl_curriculum", {})
    curriculum = RLNoiseCurriculum(
        epoch_ranges=rl_noise_cfg.get("epoch_ranges"),
        snr_levels=rl_noise_cfg.get("snr_levels"),
    )
    print(f"\n{curriculum}")

    # ── Datasets (separate train/val) ──
    train_dataset = RLSliceWiseDWIData(
        subject_list=train_subjects,
        target_H=data_cfg["target_H"],
        target_W=data_cfg["target_W"],
        target_snr=100.0,
        noise_types=noise_cfg["types"],
    )
    val_dataset = RLSliceWiseDWIData(
        subject_list=val_subjects,
        target_H=data_cfg["target_H"],
        target_W=data_cfg["target_W"],
        target_snr=100.0,   # validation at clean initially
        noise_types=noise_cfg["types"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=rl_cfg["batch_size"],
        shuffle=True, num_workers=data_cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=rl_cfg["batch_size"],
        shuffle=False, num_workers=data_cfg["num_workers"], pin_memory=True,
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

    # Initialize lazy parameters (e.g. ViT pos_emb) via a dummy forward pass
    # so that load_state_dict can match all keys from the warmup checkpoint.
    with torch.no_grad():
        dummy_dwi = torch.zeros(1, data_cfg["num_dwi_volumes"],
                                data_cfg["target_H"], data_cfg["target_W"],
                                device=device)
        dummy_mask = torch.zeros(1, 3, data_cfg["target_H"],
                                 data_cfg["target_W"], device=device)
        model(dummy_dwi, dummy_mask)
        del dummy_dwi, dummy_mask
        torch.cuda.empty_cache()

    # ── Load warm-up checkpoint ──
    if warmup_ckpt_path and os.path.exists(warmup_ckpt_path):
        ckpt = torch.load(warmup_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"✅ Loaded warm-up checkpoint: {warmup_ckpt_path}")
        print(f"   Warm-up best MAE: {ckpt.get('best_mae', 'N/A')}")
    else:
        print("⚠️  No warm-up checkpoint found. Training from scratch.")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {param_count:,}")

    # ── PPO Agent ──
    agent = PPOAgent(
        feature_dim=cfg["vit"]["embed_dim"],
        action_dim=ppo_cfg["action_dim"],
        hidden_dim=ppo_cfg["hidden_dim"],
        lr_policy=ppo_cfg["lr_policy"],
        lr_critic=ppo_cfg["lr_critic"],
        clip_eps=ppo_cfg["clip_eps"],
        ppo_epochs=ppo_cfg["ppo_epochs"],
        entropy_coeff=ppo_cfg["entropy_coeff"],
        value_loss_coeff=ppo_cfg["value_loss_coeff"],
        max_grad_norm=ppo_cfg["max_grad_norm_rl"],
        gamma=ppo_cfg["gamma"],
        device=device,
    )

    # ── Loss + Optimiser ──
    criterion = IterativeRefinementLoss(
        lambda_fw=loss_cfg["lambda_fw"],
        lambda_icvf=loss_cfg["lambda_icvf"],
        lambda_sobel=loss_cfg["lambda_sobel"],
        lambda_rl=loss_cfg["lambda_rl"],
        lambda_aux=loss_cfg["lambda_aux"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=rl_cfg["lr_model"], weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=rl_cfg["epochs"], eta_min=1e-7,
    )

    # ── Output (use HPC if accessible, else local) ──
    hpc_out = cfg["paths"]["hpc"].get("output_dir", "")
    local_out = cfg["paths"]["local"]["output_dir"]
    if hpc_out and os.path.isdir(os.path.dirname(hpc_out)):
        output_dir = hpc_out
    else:
        output_dir = local_out
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "rl_model_best.pth")

    # ── Reward weights from config ──
    rw = {
        "w_fw": reward_cfg.get("w_fw", 1.0),
        "w_icvf": reward_cfg.get("w_icvf", 1.0),
        "w_ssim": reward_cfg.get("w_ssim", 0.5),
        "w_edge": reward_cfg.get("w_edge", 0.4),
        "w_continuity": reward_cfg.get("w_continuity", 0.3),
        "w_hallucination": reward_cfg.get("w_hallucination", 0.5),
        "w_consistency": reward_cfg.get("w_consistency", 0.3),
        "w_instability": reward_cfg.get("w_instability", 0.3),
        "w_artifact": reward_cfg.get("w_artifact", 0.2),
    }

    # ── Training state ──
    best_val_mae = float("inf")
    start_epoch = 1
    rl_update_counter = 0
    metrics_history = []
    save_every = ckpt_cfg.get("save_every_n_epochs", 5)

    # Auto-resume from checkpoint if it exists
    best_model_path = os.path.join(output_dir, "rl_model_best.pth")
    rl_best_path = os.path.join(output_dir, "rl_best_model.pth")
    rl_last_path = os.path.join(output_dir, "rl_last_model.pth")

    resume_path = None
    for candidate in [rl_last_path, rl_best_path, best_model_path]:
        if os.path.exists(candidate):
            resume_path = candidate
            break

    if resume_path:
        print(f"🔄 Found existing RL checkpoint at {resume_path}. Resuming...")
        try:
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            
            # Load agent state if available
            agent_path = os.path.join(output_dir, "rl_ppo_agent_best.pth")
            if os.path.exists(agent_path):
                agent.load(agent_path)
                
            start_epoch = ckpt["epoch"] + 1
            best_val_mae = ckpt.get("best_mae", float("inf"))
            print(f"   Resuming from Epoch {start_epoch} with Best MAE {best_val_mae:.6f}")
        except Exception as e:
            print(f"⚠️  Failed to load RL checkpoint: {e}. Starting from scratch.")

    print(f"\n{'='*70}")
    print(f"RL-AUGMENTED TRAINING (PPO + Reverse Extreme-Noise Curriculum)")
    print(f"  Epochs: {rl_cfg['epochs']}")
    print(f"  RL update freq: every {rl_cfg['rl_update_freq']} batches")
    print(f"  K_outer: {iter_cfg['K_outer']}, K_inner: {cfg['pde']['K_inner']}")
    print(f"  PPO clip: {ppo_cfg['clip_eps']}, entropy: {ppo_cfg['entropy_coeff']}")
    print(f"  Train subjects: {len(train_subjects)}")
    print(f"  Val subjects: {len(val_subjects)}")
    print(f"  Curriculum: 300% → 0% (coarse-to-fine)")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, rl_cfg["epochs"] + 1):
        model.train()

        # ── Noise level from reverse curriculum ──
        target_snr = curriculum.get_target_snr(epoch)
        stage_info = curriculum.get_stage_info(epoch)
        train_dataset.set_target_snr(target_snr)

        epoch_losses = []
        epoch_fw_mae = []
        epoch_icvf_mae = []
        epoch_rewards = []
        epoch_rl_info = []
        epoch_reward_details = []

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{rl_cfg['epochs']} "
                         f"(SNR={target_snr:.1f} "
                         f"[{stage_info['description']}])")

        for batch_idx, batch in enumerate(pbar):
            (dwi_noisy, dwi_clean, gt_fw, gt_icvf,
             mask, tissue_masks, fc_prior) = [b.to(device) for b in batch]

            B = dwi_noisy.shape[0]
            K = iter_cfg["K_outer"]

            # ── Collect RL actions for each iteration ──
            model.eval()  # for RL action collection
            with torch.no_grad():
                # First: get initial ANN priors
                fw_prior, icvf_prior = model.ann_prior.predict_priors_for_batch(dwi_noisy)
                current_dwi = dwi_noisy
                current_fw = fw_prior
                current_icvf = icvf_prior

                all_actions = []
                prev_fw = fw_prior
                prev_icvf = icvf_prior

                for k in range(K):
                    x_input = torch.cat(
                        [current_dwi, current_fw, current_icvf], dim=1
                    )
                    features, x0, _ = model._single_iteration(
                        x_input, tissue_masks
                    )

                    # RL agent selects action based on features
                    actions, log_probs, values, pooled = agent.select_actions(
                        features, deterministic=False
                    )
                    all_actions.append(actions)

                    # Intermediate prediction
                    fw_k, icvf_k = model.decoder.intermediate(features)

                    # Compute per-step reward with all components
                    reward_k, reward_info = compute_reward(
                        fw_k, icvf_k, gt_fw, gt_icvf,
                        prev_fw, prev_icvf, mask, **rw,
                    )

                    # Store transition in RL buffer
                    agent.store_transition(
                        pooled, actions, log_probs, reward_k, values
                    )

                    epoch_rewards.append(reward_k.mean().item())
                    epoch_reward_details.append(reward_info)

                    # Denoise for next iteration
                    if k < K - 1:
                        denoise_str = actions[:, 0:1]
                        current_dwi = model.denoiser(
                            current_dwi, features, denoise_str
                        )
                        prev_fw = fw_k
                        prev_icvf = icvf_k
                        current_fw = fw_k
                        current_icvf = icvf_k

                # Stack actions: (B, K, 3)
                rl_actions = torch.stack(all_actions, dim=1)

            # ── Supervised forward pass (with RL actions and SNR) ──
            model.train()
            optimizer.zero_grad(set_to_none=True)

            batch_snr = torch.full((B, 1), target_snr, device=device)
            output = model(dwi_noisy, tissue_masks, rl_actions=rl_actions, snr=batch_snr, force_k_max=True)
            output["fw_final"] = torch.clamp(output["fw_final"], 0.0, 1.0)
            output["icvf_final"] = torch.clamp(output["icvf_final"], 0.0, 1.0)

            # Average reward across iterations for the loss
            avg_reward = torch.tensor(
                np.mean(epoch_rewards[-K:]), device=device
            )

            total_loss, loss_dict = criterion(
                output, gt_fw, gt_icvf, mask, reward=avg_reward
            )

            # ── Dynamic K* Target Generation & Predictor Loss ──
            with torch.no_grad():
                fw_inters = output['fw_intermediates']
                icvf_inters = output['icvf_intermediates']
                best_k_idx = torch.zeros(B, dtype=torch.long, device=device)
                min_maes = torch.full((B,), float('inf'), device=device)
                
                for k_idx, (fw_k, icvf_k) in enumerate(zip(fw_inters, icvf_inters)):
                    mae_fw = torch.abs(fw_k - gt_fw).mean(dim=(1,2,3))
                    mae_icvf = torch.abs(icvf_k - gt_icvf).mean(dim=(1,2,3))
                    mae = mae_fw + mae_icvf
                    
                    update_mask = mae < min_maes
                    min_maes[update_mask] = mae[update_mask]
                    best_k_idx[update_mask] = k_idx
                    
                # K is 1-indexed (k_idx + 1)
                k_star = (best_k_idx + 1).float()
                
            k_pred_raw = output['k_pred_raw']
            loss_k = torch.nn.functional.mse_loss(k_pred_raw, k_star)
            
            # Combine losses
            total_loss = total_loss + 0.1 * loss_k
            loss_dict['L_K_pred'] = loss_k.item()

            if not torch.isfinite(total_loss):
                print("⚠️  NaN — skipping")
                optimizer.zero_grad()
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # Free memory
            del output, total_loss
            torch.cuda.empty_cache()

            epoch_losses.append(loss_dict["total"])
            epoch_fw_mae.append(loss_dict["fw_mae"])
            epoch_icvf_mae.append(loss_dict["icvf_mae"])

            # ── PPO update ──
            rl_update_counter += 1
            if rl_update_counter % rl_cfg["rl_update_freq"] == 0:
                if agent.buffer.size > 0:
                    rl_info = agent.update(mini_batch_size=256)
                    epoch_rl_info.append(rl_info)

            pbar.set_postfix({
                "fw": f"{loss_dict['fw_mae']:.5f}",
                "icvf": f"{loss_dict['icvf_mae']:.5f}",
                "R": f"{np.mean(epoch_rewards[-K:]):.4f}",
            })

        scheduler.step()

        # ── Train epoch summary ──
        avg_fw = float(np.mean(epoch_fw_mae)) if epoch_fw_mae else float("nan")
        avg_icvf = float(np.mean(epoch_icvf_mae)) if epoch_icvf_mae else float("nan")
        avg_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        train_combined = avg_fw + avg_icvf

        rl_policy_loss = np.mean(
            [d["policy_loss"] for d in epoch_rl_info]
        ) if epoch_rl_info else 0.0

        # Average reward component details
        avg_reward_details = {}
        if epoch_reward_details:
            for key in epoch_reward_details[0].keys():
                avg_reward_details[key] = float(np.mean(
                    [d[key] for d in epoch_reward_details]
                ))

        print(f"\n  [Train] Epoch {epoch}: FW={avg_fw:.6f} ICVF={avg_icvf:.6f} "
              f"R={avg_reward:.4f} π_loss={rl_policy_loss:.4f}")
        print(f"  [Train] Target SNR: {target_snr:.1f} ({stage_info['description']})")

        # Log reward components
        if avg_reward_details:
            print(f"  [Reward] SSIM={avg_reward_details.get('ssim_reward', 0):.4f} "
                  f"Edge={avg_reward_details.get('edge_reward', 0):.4f} "
                  f"Halluc={avg_reward_details.get('hallucination_penalty', 0):.4f} "
                  f"Consist={avg_reward_details.get('consistency_reward', 0):.4f}")

        # ── Validation ──
        print(f"  [Val - Clean] Running validation on {len(val_subjects)} subjects...")
        val_dataset.set_target_snr(100.0)
        val_metrics_clean = run_validation(
            model, val_loader, device, epoch, output_dir,
            save_predictions=val_cfg.get("save_predictions", True),
            save_visualizations=val_cfg.get("save_visualizations", True),
            prefix="clean_"
        )

        print(f"  [Val - Clean] FW_MAE={val_metrics_clean['fw_mae']:.6f} "
              f"ICVF_MAE={val_metrics_clean['icvf_mae']:.6f} "
              f"Combined={val_metrics_clean['combined_mae']:.6f} "
              f"SSIM={val_metrics_clean['ssim_combined']:.4f}")

        print(f"  [Val - Noisy] Running validation at SNR={target_snr:.1f}...")
        val_dataset.set_target_snr(target_snr)
        val_metrics_noisy = run_validation(
            model, val_loader, device, epoch, output_dir,
            save_predictions=val_cfg.get("save_predictions", True),
            save_visualizations=val_cfg.get("save_visualizations", True),
            prefix="noisy_"
        )

        print(f"  [Val - Noisy] FW_MAE={val_metrics_noisy['fw_mae']:.6f} "
              f"ICVF_MAE={val_metrics_noisy['icvf_mae']:.6f} "
              f"Combined={val_metrics_noisy['combined_mae']:.6f} "
              f"SSIM={val_metrics_noisy['ssim_combined']:.4f}")

        # Metrics logging
        val_metrics = val_metrics_clean
        epoch_metrics = {
            "epoch": epoch,
            "target_snr": target_snr,
            "noise_stage": stage_info["description"],
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "train_fw_mae": avg_fw,
            "train_icvf_mae": avg_icvf,
            "train_combined_mae": train_combined,
            "train_reward": avg_reward,
            "rl_policy_loss": rl_policy_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"val_clean_{k}": v for k, v in val_metrics_clean.items() if k != "epoch"},
            **{f"val_noisy_{k}": v for k, v in val_metrics_noisy.items() if k != "epoch"},
            **{f"reward_{k}": v for k, v in avg_reward_details.items()},
        }
        metrics_history.append(epoch_metrics)
        save_metrics_log(metrics_history, output_dir, prefix="rl_")

        # ── Checkpointing (validation-driven) ──
        val_combined = val_metrics["combined_mae"]
        is_best = val_combined < best_val_mae

        if is_best:
            best_val_mae = val_combined
            print(f"  ✅ New best validation MAE: {best_val_mae:.6f}")

        save_checkpoint(
            model, optimizer, epoch, val_metrics, output_dir,
            is_best=is_best, scheduler=scheduler, agent=agent,
            config=cfg, prefix="rl_",
        )

        # Also save to legacy path for backward compatibility
        if is_best:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mae": best_val_mae,
                "config": cfg,
            }, model_path)
            agent.save(os.path.join(output_dir, "ppo_agent_best.pth"))

        # Periodic checkpoint
        save_periodic_checkpoint(
            model, optimizer, epoch, val_metrics, output_dir,
            save_every=save_every, scheduler=scheduler, agent=agent,
            config=cfg, prefix="rl_",
        )

    print(f"\n{'='*70}")
    print(f"RL training complete! Best validation MAE: {best_val_mae:.6f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"{'='*70}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RL-augmented training")
    parser.add_argument("--warmup-ckpt", type=str, default=None,
                        help="Path to warm-up checkpoint (warmup_best.pth)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    cfg = load_config(args.config)
    subjects = discover_subjects(cfg)

    if not subjects:
        print("[ERROR] No subjects found.")
        return

    print(f"\n✅ Found {len(subjects)} subjects for RL training.")

    # Auto-discover warmup checkpoint
    warmup_path = args.warmup_ckpt
    if warmup_path is None:
        # Try local first, then HPC — check multiple naming patterns
        for candidate_dir in [cfg["paths"]["local"]["output_dir"],
                               cfg["paths"]["hpc"].get("output_dir", "")]:
            for name in ["warmup_best_model.pth", "warmup_best.pth"]:
                candidate = os.path.join(candidate_dir, name)
                if os.path.exists(candidate):
                    warmup_path = candidate
                    break
            if warmup_path:
                break

    train_rl(cfg, subjects, warmup_ckpt_path=warmup_path)


if __name__ == "__main__":
    main()
