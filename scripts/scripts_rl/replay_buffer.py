"""Replay buffer for TD-Learning Diffusion Policy (Plan A).

Stores transitions from online rollout and demonstration data.
Supports mixed sampling (online + demo with configurable ratio).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class Transition:
    """Single transition for the replay buffer.

    All values are stored as numpy arrays to save GPU memory.
    They are converted to tensors on sampling.
    """
    table_img: np.ndarray     # (3, H, W) float32 [0,1]
    wrist_img: np.ndarray     # (3, H, W) float32 [0,1]
    state: np.ndarray         # (15,)
    action: np.ndarray        # (horizon, 8) or (8,) depending on chunking
    reward: float
    next_table_img: np.ndarray
    next_wrist_img: np.ndarray
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size replay buffer with efficient ring-buffer storage."""

    def __init__(
        self,
        capacity: int,
        state_dim: int = 15,
        action_dim: int = 8,
        action_horizon: int = 16,
        image_shape: tuple = (3, 128, 128),
    ):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

        # Pre-allocate numpy arrays
        self.table_imgs = np.zeros((capacity, *image_shape), dtype=np.float32)
        self.wrist_imgs = np.zeros((capacity, *image_shape), dtype=np.float32)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_horizon, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_table_imgs = np.zeros((capacity, *image_shape), dtype=np.float32)
        self.next_wrist_imgs = np.zeros((capacity, *image_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        table_img: np.ndarray,
        wrist_img: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_table_img: np.ndarray,
        next_wrist_img: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition."""
        idx = self.ptr
        self.table_imgs[idx] = table_img
        self.wrist_imgs[idx] = wrist_img
        self.states[idx] = state
        if action.ndim == 1:
            self.actions[idx, 0] = action
        else:
            self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_table_imgs[idx] = next_table_img
        self.next_wrist_imgs[idx] = next_wrist_img
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        table_imgs: np.ndarray,
        wrist_imgs: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_table_imgs: np.ndarray,
        next_wrist_imgs: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add a batch of transitions."""
        n = len(rewards)
        for i in range(n):
            act = actions[i] if actions.ndim >= 2 else actions[i:i+1]
            self.add(
                table_imgs[i], wrist_imgs[i], states[i], act,
                float(rewards[i]),
                next_table_imgs[i], next_wrist_imgs[i], next_states[i],
                bool(dones[i]),
            )

    def sample(self, batch_size: int, device: str = "cuda") -> Dict[str, torch.Tensor]:
        """Sample a random batch."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "table_img": torch.from_numpy(self.table_imgs[idxs]).to(device),
            "wrist_img": torch.from_numpy(self.wrist_imgs[idxs]).to(device),
            "state": torch.from_numpy(self.states[idxs]).to(device),
            "action": torch.from_numpy(self.actions[idxs]).to(device),
            "reward": torch.from_numpy(self.rewards[idxs]).to(device),
            "next_table_img": torch.from_numpy(self.next_table_imgs[idxs]).to(device),
            "next_wrist_img": torch.from_numpy(self.next_wrist_imgs[idxs]).to(device),
            "next_state": torch.from_numpy(self.next_states[idxs]).to(device),
            "done": torch.from_numpy(self.dones[idxs]).to(device),
        }

    def __len__(self) -> int:
        return self.size


class DemoBuffer:
    """Buffer for demonstration data (loaded from LeRobot dataset or NPZ).

    Read-only — data is loaded once and sampled repeatedly.
    """

    def __init__(self, device: str = "cuda"):
        self.table_imgs: list[np.ndarray] = []
        self.wrist_imgs: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.device = device
        self._finalized = False

    def add_episode(
        self,
        table_imgs: np.ndarray,
        wrist_imgs: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        """Add all frames from a demo episode."""
        T = len(states)
        for t in range(T):
            self.table_imgs.append(table_imgs[t])
            self.wrist_imgs.append(wrist_imgs[t])
            self.states.append(states[t])
            self.actions.append(actions[t])

    def finalize(self) -> None:
        """Stack lists into contiguous arrays for efficient sampling."""
        self._table_imgs = np.stack(self.table_imgs, axis=0)
        self._wrist_imgs = np.stack(self.wrist_imgs, axis=0)
        self._states = np.stack(self.states, axis=0)
        self._actions = np.stack(self.actions, axis=0)
        # Free original lists
        self.table_imgs = []
        self.wrist_imgs = []
        self.states = []
        self.actions = []
        self._finalized = True

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch of demo transitions."""
        if not self._finalized:
            self.finalize()
        n = len(self._states)
        idxs = np.random.randint(0, n, size=batch_size)
        return {
            "table_img": torch.from_numpy(self._table_imgs[idxs]).to(self.device),
            "wrist_img": torch.from_numpy(self._wrist_imgs[idxs]).to(self.device),
            "state": torch.from_numpy(self._states[idxs]).to(self.device),
            "action": torch.from_numpy(self._actions[idxs]).to(self.device),
        }

    def __len__(self) -> int:
        if self._finalized:
            return len(self._states)
        return len(self.states)


class MixedReplayBuffer:
    """Combines online replay buffer and demo buffer with configurable ratio."""

    def __init__(
        self,
        online_buffer: ReplayBuffer,
        demo_buffer: DemoBuffer,
        demo_ratio: float = 0.25,
    ):
        self.online = online_buffer
        self.demo = demo_buffer
        self.demo_ratio = demo_ratio

    def sample(
        self, batch_size: int, device: str = "cuda",
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Sample from both buffers.

        Returns:
            (online_batch, demo_batch) — caller handles combining them.
        """
        n_demo = max(1, int(batch_size * self.demo_ratio))
        n_online = batch_size - n_demo

        online_batch = self.online.sample(n_online, device=device)
        demo_batch = self.demo.sample(n_demo)
        return online_batch, demo_batch
