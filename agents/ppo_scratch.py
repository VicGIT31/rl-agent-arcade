"""Proximal Policy Optimization, implemented from scratch.

This is the core of the project. Nothing here is delegated to an RL library —
the rollout collection, Generalized Advantage Estimation, the clipped surrogate
objective, the value loss and the entropy bonus are all written by hand so the
maths is explicit and auditable.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017).
The clipped objective is

    L_CLIP(theta) = E[ min( r_t(theta) * A_t,
                            clip(r_t(theta), 1 - eps, 1 + eps) * A_t ) ]

where r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from agents.networks import ActorCritic


@dataclass
class PPOConfig:
    obs_dim: int
    n_actions: int
    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.99          # discount factor
    gae_lambda: float = 0.95     # GAE smoothing
    clip_eps: float = 0.2        # PPO clipping range
    entropy_coef: float = 0.01   # exploration bonus weight
    value_coef: float = 0.5      # value-loss weight
    max_grad_norm: float = 0.5   # gradient clipping
    update_epochs: int = 4       # passes over each rollout
    minibatch_size: int = 256
    rollout_steps: int = 2048    # transitions collected per update
    device: str = "cpu"
    seed: int = 0


class RolloutBuffer:
    """Fixed-size storage for one on-policy rollout."""

    def __init__(self, size: int, obs_dim: int, device: str) -> None:
        self.size = size
        self.device = device
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, done, value) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.size

    def reset(self) -> None:
        self.ptr = 0

    def compute_gae(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generalized Advantage Estimation (Schulman et al., 2016).

        Walks backward through the rollout accumulating

            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
            A_t     = delta_t + gamma * lambda * (1 - done) * A_{t+1}

        Returns (advantages, returns) where returns = advantages + values are
        the value-function regression targets.
        """
        advantages = np.zeros(self.size, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.size)):
            next_value = last_value if t == self.size - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return advantages, returns


class PPO:
    def __init__(self, config: PPOConfig) -> None:
        self.cfg = config
        torch.manual_seed(config.seed)
        self.device = torch.device(config.device)
        self.policy = ActorCritic(
            config.obs_dim, config.n_actions, config.hidden_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr, eps=1e-5)
        self.buffer = RolloutBuffer(config.rollout_steps, config.obs_dim, config.device)

    # ---------------------------------------------------------------- acting

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob, value = self.policy.act(obs_t)
        return int(action.item()), float(log_prob.item()), float(value.item())

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, value = self.policy(obs_t)
        return float(value.item())

    @torch.no_grad()
    def act_greedy(self, obs: np.ndarray) -> int:
        """Deterministic action (argmax) for evaluation / demo playback."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist, _ = self.policy(obs_t)
        return int(dist.probs.argmax(dim=-1).item())

    # ---------------------------------------------------------------- update

    def update(self, last_value: float) -> dict:
        cfg = self.cfg
        advantages, returns = self.buffer.compute_gae(
            last_value, cfg.gamma, cfg.gae_lambda
        )

        obs = torch.as_tensor(self.buffer.obs, device=self.device)
        actions = torch.as_tensor(self.buffer.actions, device=self.device)
        old_log_probs = torch.as_tensor(self.buffer.log_probs, device=self.device)
        returns_t = torch.as_tensor(returns, device=self.device)
        adv_t = torch.as_tensor(advantages, device=self.device)
        # Advantage normalisation reduces gradient variance across updates.
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = cfg.rollout_steps
        indices = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
        n_batches = 0

        for _ in range(cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, cfg.minibatch_size):
                mb = indices[start : start + cfg.minibatch_size]
                mb_obs = obs[mb]
                mb_actions = actions[mb]

                new_log_probs, entropy, values = self.policy.evaluate_actions(
                    mb_obs, mb_actions
                )

                # Importance-sampling ratio r_t(theta).
                ratio = torch.exp(new_log_probs - old_log_probs[mb])
                mb_adv = adv_t[mb]

                # Clipped surrogate objective (negated -> a loss to minimise).
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value regression toward the GAE returns.
                value_loss = nn.functional.mse_loss(values, returns_t[mb])

                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy_bonus.item()
                stats["clip_frac"] += clip_frac.item()
                n_batches += 1

        for k in stats:
            stats[k] /= max(n_batches, 1)
        self.buffer.reset()
        return stats

    # ------------------------------------------------------------ persistence

    def save(self, path: str) -> None:
        torch.save(
            {"model_state": self.policy.state_dict(), "config": self.cfg.__dict__},
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PPO":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = PPOConfig(**ckpt["config"])
        cfg.device = device
        agent = cls(cfg)
        agent.policy.load_state_dict(ckpt["model_state"])
        agent.policy.eval()
        return agent
