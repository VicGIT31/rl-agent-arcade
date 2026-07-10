"""Quantitative evaluation of a trained checkpoint.

Runs N episodes with the greedy policy and reports mean/std/max game score and
episode reward. Also reports a random-agent baseline for context so the
improvement is quantified, not just asserted.

Usage:
    python -m evaluation.evaluate --checkpoint checkpoints/best_model.pt --episodes 100
"""

from __future__ import annotations

import argparse

import numpy as np

from envs.snake_env import SnakeEnv
from agents.ppo_scratch import PPO


def run_policy(policy_fn, env: SnakeEnv, episodes: int, seed: int = 0) -> dict:
    scores, rewards, lengths = [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        total_r, steps = 0.0, 0
        while not done:
            action = policy_fn(obs)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            steps += 1
            done = terminated or truncated
        scores.append(info["score"])
        rewards.append(total_r)
        lengths.append(steps)
    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "max_score": int(np.max(scores)),
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=12)
    args = parser.parse_args()

    env = SnakeEnv(grid_size=args.grid_size)

    rng = np.random.default_rng(0)
    random_stats = run_policy(
        lambda obs: int(rng.integers(env.action_space.n)), env, args.episodes
    )

    agent = PPO.load(args.checkpoint)
    agent_stats = run_policy(agent.act_greedy, env, args.episodes)

    print(f"\nEvaluation over {args.episodes} episodes (grid {args.grid_size}x{args.grid_size})")
    print("-" * 56)
    print(f"{'metric':<16}{'random':>12}{'PPO (scratch)':>18}")
    print("-" * 56)
    print(f"{'mean score':<16}{random_stats['mean_score']:>12.2f}"
          f"{agent_stats['mean_score']:>18.2f}")
    print(f"{'max score':<16}{random_stats['max_score']:>12d}"
          f"{agent_stats['max_score']:>18d}")
    print(f"{'mean reward':<16}{random_stats['mean_reward']:>12.2f}"
          f"{agent_stats['mean_reward']:>18.2f}")
    print(f"{'mean length':<16}{random_stats['mean_length']:>12.1f}"
          f"{agent_stats['mean_length']:>18.1f}")
    print("-" * 56)
    improvement = agent_stats["mean_score"] - random_stats["mean_score"]
    print(f"Mean-score improvement over random: +{improvement:.2f}")


if __name__ == "__main__":
    main()
