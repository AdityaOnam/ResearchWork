#!/usr/bin/env python3
"""
rl_agent.py

PPO (Proximal Policy Optimisation) agent for the iterative refinement loop.

Responsibilities:
  1. Collect trajectories from the iterative model's refinement steps.
  2. Compute rewards using the composite reward function.
  3. Update the Actor-Critic policy using the clipped PPO objective.

Usage:
  agent = PPOAgent(policy, ...)
  # During training loop:
  actions = agent.select_actions(features)
  # After collecting rewards:
  agent.update(buffer)
"""

import torch
import torch.nn as nn

from rl.policy import ActorCritic
from rl.replay_buffer import TrajectoryBuffer


class PPOAgent:
    """PPO agent managing policy updates for the RL refinement controller.

    Parameters
    ----------
    feature_dim : int
        Input feature dimension for the policy (default 64).
    action_dim : int
        Number of continuous action dimensions (default 3).
    hidden_dim : int
        Hidden layer width in Actor-Critic (default 128).
    lr_policy : float
        Learning rate for the policy network (default 3e-4).
    lr_critic : float
        Learning rate for the critic (shared optimizer, but can be split).
    clip_eps : float
        PPO clipping epsilon (default 0.2).
    ppo_epochs : int
        Number of PPO update epochs per batch of trajectories (default 4).
    entropy_coeff : float
        Entropy bonus coefficient for exploration (default 0.01).
    value_loss_coeff : float
        Critic loss coefficient (default 0.5).
    max_grad_norm : float
        Gradient clipping norm (default 0.5).
    gamma : float
        Discount factor (default 0.99).
    gae_lambda : float
        GAE lambda (default 0.95).
    buffer_capacity : int
        Trajectory buffer size (default 2048).
    device : torch.device
        Compute device.
    """

    def __init__(self,
                 feature_dim: int = 64,
                 action_dim: int = 3,
                 hidden_dim: int = 128,
                 lr_policy: float = 3e-4,
                 lr_critic: float = 1e-3,
                 clip_eps: float = 0.2,
                 ppo_epochs: int = 4,
                 entropy_coeff: float = 0.01,
                 value_loss_coeff: float = 0.5,
                 max_grad_norm: float = 0.5,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 buffer_capacity: int = 2048,
                 device: torch.device = None):

        self.device = device or torch.device("cpu")
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm

        # Policy network
        self.policy = ActorCritic(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        ).to(self.device)

        # Optimizer (single optimizer for both actor and critic)
        self.optimizer = torch.optim.Adam([
            {"params": self.policy.shared.parameters(), "lr": lr_policy},
            {"params": self.policy.actor_alpha.parameters(), "lr": lr_policy},
            {"params": self.policy.actor_beta.parameters(), "lr": lr_policy},
            {"params": self.policy.critic.parameters(), "lr": lr_critic},
        ])

        # Trajectory buffer
        self.buffer = TrajectoryBuffer(
            capacity=buffer_capacity,
            feature_dim=feature_dim,
            action_dim=action_dim,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=self.device,
        )

    # ------------------------------------------------------------------
    def select_actions(self,
                       feature_map: torch.Tensor,
                       deterministic: bool = False) -> tuple:
        """Select actions for a batch of feature maps.

        Args:
            feature_map: (B, C, H, W) — feature map from the iterative model.
            deterministic: If True, use mean action (for evaluation).

        Returns:
            actions:  (B, action_dim) in [0, 1].
            log_probs: (B,)
            values:   (B, 1)
        """
        # Spatial pooling → (B, C)
        features = ActorCritic.pool_features(feature_map)

        with torch.set_grad_enabled(not deterministic):
            actions, log_probs, values = self.policy.get_action(
                features, deterministic=deterministic
            )

        return actions, log_probs, values, features

    # ------------------------------------------------------------------
    def store_transition(self,
                         features: torch.Tensor,
                         actions: torch.Tensor,
                         log_probs: torch.Tensor,
                         rewards: torch.Tensor,
                         values: torch.Tensor,
                         dones: torch.Tensor = None):
        """Store a batch of transitions in the replay buffer.

        Args:
            All tensors are (B,) or (B, dim).
            dones: If None, all are treated as non-terminal (0).
        """
        if dones is None:
            dones = torch.zeros(features.shape[0], device=self.device)

        self.buffer.store(features, actions, log_probs,
                          rewards, values, dones)

    # ------------------------------------------------------------------
    def update(self, mini_batch_size: int = 256) -> dict:
        """Run PPO update using stored trajectories.

        Call after a full rollout (fill buffer → compute advantages → update).

        Args:
            mini_batch_size: Size of random mini-batches for gradient steps.

        Returns:
            dict with mean losses for logging.
        """
        self.buffer.compute_advantages()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(mini_batch_size):
                states = batch["states"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Evaluate current policy on stored states/actions
                new_log_probs, entropy, values = self.policy.evaluate_action(
                    states, actions
                )

                # PPO clipped surrogate objective
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps,
                                1.0 + self.clip_eps)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                value_loss = nn.functional.mse_loss(
                    values.squeeze(-1), returns
                )

                # Entropy bonus (encourage exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coeff * value_loss
                    + self.entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss).item()
                n_updates += 1

        # Reset buffer for next rollout
        self.buffer.reset()

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save policy checkpoint."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load policy checkpoint."""
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck["policy_state_dict"])
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        print(f"✅ Loaded PPO agent from {path}")
