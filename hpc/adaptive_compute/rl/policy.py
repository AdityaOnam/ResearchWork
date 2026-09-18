#!/usr/bin/env python3
"""
policy.py

Actor-Critic policy network for PPO-based reinforcement learning.

The policy operates on spatially-pooled feature maps from the iterative model
and outputs continuous actions that control:
  1. Denoising strength    (how aggressively to filter)   [0, 1]
  2. Noise adaptation      (next iteration noise level)   [0, 1]
  3. PDE intensity         (scaling for PDE update step)  [0, 1]

Architecture:
  Shared MLP → Actor head (3 actions, Beta distribution)
              → Critic head (1 value)
"""

import torch
import torch.nn as nn
from torch.distributions import Beta


class ActorCritic(nn.Module):
    """PPO Actor-Critic for the RL refinement controller.

    Parameters
    ----------
    feature_dim : int
        Dimension of the input feature vector (from spatial pooling
        of the ViT feature map). Default 64.
    hidden_dim : int
        Hidden layer width. Default 128.
    action_dim : int
        Number of continuous action dimensions. Default 3.
    """

    def __init__(self, feature_dim: int = 64,
                 hidden_dim: int = 128,
                 action_dim: int = 3):
        super().__init__()
        self.action_dim = action_dim

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor: outputs alpha and beta parameters for Beta distribution
        # (ensures actions are bounded in [0, 1])
        self.actor_alpha = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus(),   # alpha > 0
        )
        self.actor_beta = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus(),   # beta > 0
        )

        # Critic: state value estimate
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    # ------------------------------------------------------------------
    def forward(self, features: torch.Tensor) -> tuple:
        """Compute action distribution and value for given features.

        Args:
            features: (B, feature_dim)

        Returns:
            dist:   Beta distribution over actions (B, action_dim).
            value:  (B, 1) state value estimate.
        """
        shared = self.shared(features)

        alpha = self.actor_alpha(shared) + 1.0   # shift to alpha >= 1
        beta = self.actor_beta(shared) + 1.0     # shift to beta >= 1

        dist = Beta(alpha, beta)
        value = self.critic(shared)

        return dist, value

    # ------------------------------------------------------------------
    def get_action(self,
                   features: torch.Tensor,
                   deterministic: bool = False) -> tuple:
        """Sample an action from the policy.

        Args:
            features: (B, feature_dim)
            deterministic: If True, return the mean action.

        Returns:
            action:    (B, action_dim) in [0, 1].
            log_prob:  (B,) log probability of the action.
            value:     (B, 1) critic value estimate.
        """
        dist, value = self.forward(features)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        # Sum log probs across action dimensions
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value

    # ------------------------------------------------------------------
    def evaluate_action(self,
                        features: torch.Tensor,
                        action: torch.Tensor) -> tuple:
        """Evaluate log probability and entropy for a given action.

        Used during PPO updates with stored trajectories.

        Args:
            features: (B, feature_dim)
            action: (B, action_dim) in [0, 1].

        Returns:
            log_prob: (B,)
            entropy:  (B,) entropy of the action distribution.
            value:    (B, 1) critic value estimate.
        """
        dist, value = self.forward(features)

        # Clamp actions to valid Beta range to avoid log(0)
        action = action.clamp(1e-6, 1.0 - 1e-6)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, value

    # ------------------------------------------------------------------
    @staticmethod
    def pool_features(feature_map: torch.Tensor) -> torch.Tensor:
        """Spatially pool a feature map to a 1-D vector for the policy.

        Args:
            feature_map: (B, C, H, W)

        Returns:
            (B, C) — global average pooled features.
        """
        return feature_map.mean(dim=(2, 3))
