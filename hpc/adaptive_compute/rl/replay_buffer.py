#!/usr/bin/env python3
"""
replay_buffer.py

Trajectory storage for PPO training.

Stores transitions from the iterative refinement loop:
  (state_features, action, log_prob, reward, value, done)

Uses generalised advantage estimation (GAE) for computing advantages.
"""

import torch
import numpy as np


class TrajectoryBuffer:
    """Fixed-size buffer for storing PPO rollout trajectories.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store before an update.
    feature_dim : int
        Dimension of the state feature vector (default 64).
    action_dim : int
        Dimension of the action vector (default 3).
    gamma : float
        Discount factor (default 0.99).
    gae_lambda : float
        GAE lambda for advantage estimation (default 0.95).
    device : torch.device
        Target device for tensors.
    """

    def __init__(self,
                 capacity: int = 2048,
                 feature_dim: int = 64,
                 action_dim: int = 3,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 device: torch.device = None):

        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device or torch.device("cpu")

        # Pre-allocate tensors
        self.states = torch.zeros(capacity, feature_dim, device=self.device)
        self.actions = torch.zeros(capacity, action_dim, device=self.device)
        self.log_probs = torch.zeros(capacity, device=self.device)
        self.rewards = torch.zeros(capacity, device=self.device)
        self.values = torch.zeros(capacity, device=self.device)
        self.dones = torch.zeros(capacity, device=self.device)

        # Computed after rollout
        self.advantages = torch.zeros(capacity, device=self.device)
        self.returns = torch.zeros(capacity, device=self.device)

        self.ptr = 0
        self.size = 0

    # ------------------------------------------------------------------
    def store(self,
              state: torch.Tensor,
              action: torch.Tensor,
              log_prob: torch.Tensor,
              reward: torch.Tensor,
              value: torch.Tensor,
              done: torch.Tensor):
        """Store a batch of transitions.

        Args:
            All tensors should be (B,) or (B, dim) and will be stored
            sequentially in the buffer.
        """
        B = state.shape[0]
        if self.ptr + B > self.capacity:
            # Wrap around or truncate
            B = self.capacity - self.ptr

        end = self.ptr + B
        self.states[self.ptr:end] = state[:B].detach()
        self.actions[self.ptr:end] = action[:B].detach()
        self.log_probs[self.ptr:end] = log_prob[:B].detach()
        self.rewards[self.ptr:end] = reward[:B].detach()
        self.values[self.ptr:end] = value[:B].squeeze(-1).detach()
        self.dones[self.ptr:end] = done[:B].detach()

        self.ptr = end
        self.size = min(self.size + B, self.capacity)

    # ------------------------------------------------------------------
    def compute_advantages(self, last_value: float = 0.0):
        """Compute GAE advantages and discounted returns.

        Call this after a full rollout, before the PPO update.

        Args:
            last_value: Value estimate for the state after the last stored
                        transition (bootstrap value).
        """
        n = self.size
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_val = last_value
                next_done = 0.0
            else:
                next_val = self.values[t + 1].item()
                next_done = self.dones[t + 1].item()

            delta = (
                self.rewards[t]
                + self.gamma * next_val * (1.0 - next_done)
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * gae
            self.advantages[t] = gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

        # Normalise advantages
        adv = self.advantages[:n]
        self.advantages[:n] = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ------------------------------------------------------------------
    def get_batches(self, batch_size: int = 256):
        """Yield random mini-batches from the buffer.

        Args:
            batch_size: Number of transitions per mini-batch.

        Yields:
            dict with keys: states, actions, log_probs, advantages, returns.
        """
        indices = np.random.permutation(self.size)
        for start in range(0, self.size, batch_size):
            end = min(start + batch_size, self.size)
            idx = indices[start:end]
            idx_t = torch.tensor(idx, dtype=torch.long, device=self.device)

            yield {
                "states": self.states[idx_t],
                "actions": self.actions[idx_t],
                "log_probs": self.log_probs[idx_t],
                "advantages": self.advantages[idx_t],
                "returns": self.returns[idx_t],
            }

    # ------------------------------------------------------------------
    def reset(self):
        """Reset buffer for the next rollout."""
        self.ptr = 0
        self.size = 0
