"""Stable-Baselines3 PPO baseline.

Purpose: an objective reference point. If SB3's battle-tested PPO learns this
environment well, then the environment is sound and any failure of the
from-scratch agent is an implementation bug rather than a broken task. The
README reports the from-scratch score as a percentage of this baseline.

This module intentionally uses SB3 as a black box -- it is the *comparison*,
not the contribution.

Usage:
    python -m agents.baseline_sb3 --timesteps 300000
"""

from __future__ import annotations

import argparse

import numpy as np


def make_env(grid_size: int):
    from envs.snake_env import SnakeEnv
    return SnakeEnv(grid_size=grid_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--save", default="checkpoints/sb3_ppo.zip")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO as SB3PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError:
        raise SystemExit(
            "stable-baselines3 is not installed. Run: pip install stable-baselines3"
        )

    vec_env = make_vec_env(lambda: make_env(args.grid_size), n_envs=4)
    model = SB3PPO("MlpPolicy", vec_env, verbose=1, seed=0)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)

    # Evaluate greedily.
    env = make_env(args.grid_size)
    scores = []
    for i in range(args.episodes):
        obs, _ = env.reset(seed=i)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        scores.append(info["score"])
    print(f"\nSB3 PPO baseline over {args.episodes} episodes: "
          f"mean score {np.mean(scores):.2f} (max {np.max(scores)})")


if __name__ == "__main__":
    main()
