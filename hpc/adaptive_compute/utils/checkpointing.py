#!/usr/bin/env python3
"""
utils/checkpointing.py

Checkpoint management for the RL iterative refinement model.

Supports:
  ✅ best_model.pth  — best validation MAE
  ✅ last_model.pth  — most recent epoch
  ✅ checkpoint_epoch_N.pth  — periodic (every 5 epochs)
  ✅ Optional PPO agent state saving
"""

import os
import json
import torch


def save_checkpoint(model, optimizer, epoch, metrics, output_dir,
                    is_best=False, scheduler=None, agent=None,
                    config=None, prefix=""):
    """Save model checkpoint with metadata.

    Args:
        model: nn.Module — the model to save.
        optimizer: torch.optim.Optimizer.
        epoch: int — current epoch number.
        metrics: dict — validation metrics (fw_mae, icvf_mae, etc.)
        output_dir: str — directory to save checkpoints.
        is_best: bool — if True, save as best_model.pth.
        scheduler: optional LR scheduler.
        agent: optional PPOAgent.
        config: optional dict — training config.
        prefix: optional prefix (e.g. 'warmup_' or 'rl_').
    """
    os.makedirs(output_dir, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    if config is not None:
        ckpt["config"] = config

    # Add best_mae for backward compatibility
    combined_mae = metrics.get("combined_mae", float("inf"))
    ckpt["best_mae"] = combined_mae

    # ── Always save last ──
    last_path = os.path.join(output_dir, f"{prefix}last_model.pth")
    torch.save(ckpt, last_path)

    # ── Save best if improved ──
    if is_best:
        best_path = os.path.join(output_dir, f"{prefix}best_model.pth")
        torch.save(ckpt, best_path)

        # Also save PPO agent if provided
        if agent is not None:
            agent_path = os.path.join(output_dir, f"{prefix}ppo_agent_best.pth")
            agent.save(agent_path)

    return last_path


def save_periodic_checkpoint(model, optimizer, epoch, metrics, output_dir,
                              save_every=5, scheduler=None, agent=None,
                              config=None, prefix=""):
    """Save periodic checkpoint every N epochs.

    Args:
        save_every: int — save every N epochs (default 5).
        Other args same as save_checkpoint.
    """
    if epoch % save_every != 0:
        return None

    os.makedirs(output_dir, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    if config is not None:
        ckpt["config"] = config

    ckpt_path = os.path.join(output_dir, f"{prefix}checkpoint_epoch_{epoch}.pth")
    torch.save(ckpt, ckpt_path)

    # Also save PPO agent at periodic checkpoints
    if agent is not None:
        agent_path = os.path.join(output_dir, f"{prefix}ppo_agent_epoch_{epoch}.pth")
        agent.save(agent_path)

    print(f"  💾 Periodic checkpoint saved: {ckpt_path}")
    return ckpt_path


def save_metrics_log(metrics_history, output_dir, prefix=""):
    """Save epoch-wise metrics to a JSON file.

    Args:
        metrics_history: list[dict] — one dict per epoch with metric values.
        output_dir: str — directory to save.
        prefix: optional prefix (e.g. 'warmup_' or 'rl_').
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{prefix}metrics_log.json")

    with open(log_path, "w") as f:
        json.dump(metrics_history, f, indent=2)

    return log_path


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None,
                    device=None):
    """Load a checkpoint and optionally restore optimizer/scheduler state.

    Args:
        model: nn.Module to load weights into.
        checkpoint_path: str — path to .pth file.
        optimizer: optional optimizer to restore.
        scheduler: optional scheduler to restore.
        device: torch.device.

    Returns:
        dict with 'epoch', 'metrics', 'best_mae' from checkpoint.
    """
    if device is None:
        device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return {
        "epoch": ckpt.get("epoch", 0),
        "metrics": ckpt.get("metrics", {}),
        "best_mae": ckpt.get("best_mae", float("inf")),
    }
