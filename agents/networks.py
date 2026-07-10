"""Actor-critic networks for the from-scratch PPO agent.

A shared MLP trunk feeds two heads:

* the **policy head** (actor) outputs logits over the discrete action space;
* the **value head** (critic) outputs a scalar state-value estimate.

Sharing the trunk keeps the model small and lets both heads benefit from the
same learned features — a common choice for low-dimensional observations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    """Orthogonal weight init + zero bias.

    Orthogonal initialisation is the standard trick in PPO implementations: it
    keeps activations well-scaled through the network and noticeably stabilises
    early training compared to the default PyTorch init.
    """
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden_dim), gain=2**0.5),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=2**0.5),
            nn.Tanh(),
        )
        # Small gain on the policy head keeps the initial policy close to uniform.
        self.policy_head = _orthogonal_init(nn.Linear(hidden_dim, n_actions), gain=0.01)
        self.value_head = _orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        features = self.trunk(obs)
        logits = self.policy_head(features)
        dist = Categorical(logits=logits)
        value = self.value_head(features).squeeze(-1)
        return dist, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action. Returns (action, log_prob, value)."""
        dist, value = self.forward(obs)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For a batch of (obs, action): log-probs, entropy, values.

        Used during the PPO update to recompute log-probs under the *current*
        policy so the importance-sampling ratio can be formed.
        """
        dist, value = self.forward(obs)
        return dist.log_prob(actions), dist.entropy(), value
