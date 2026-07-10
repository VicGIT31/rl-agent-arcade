"""Render an agent playing Snake and export it as a GIF.

Used to build the progression story (random -> mid-training -> expert) by
pointing ``--checkpoint`` at checkpoints saved at different training stages.
Pass ``--random`` for the untrained baseline GIF.

Usage:
    python -m evaluation.record_video --random --out media/random_agent.gif
    python -m evaluation.record_video --checkpoint checkpoints/ckpt_update_00025.pt \
        --out media/mid_training_agent.gif
    python -m evaluation.record_video --checkpoint checkpoints/best_model.pt \
        --out media/expert_agent.gif
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import imageio.v2 as imageio

from envs.snake_env import SnakeEnv
from agents.ppo_scratch import PPO


def record(policy_fn, env: SnakeEnv, max_frames: int, seed: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    frames.append(env.render())
    for _ in range(max_frames):
        action = policy_fn(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            # Pause on the final frame so the GIF is readable.
            frames.extend([frames[-1]] * 8)
            break
    return frames


def pick_best_episode(policy_fn, env, seeds, max_frames):
    """Try several seeds and keep the episode with the highest score (best demo)."""
    best_frames, best_score = None, -1
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        frames = [env.render()]
        score = 0
        for _ in range(max_frames):
            action = policy_fn(obs)
            obs, _, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            score = info["score"]
            if terminated or truncated:
                frames.extend([frames[-1]] * 8)
                break
        if score > best_score:
            best_frames, best_score = frames, score
    return best_frames, best_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--random", action="store_true",
                        help="Record an untrained random agent.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--best-of", type=int, default=8,
                        help="Record this many episodes and keep the best-scoring one.")
    args = parser.parse_args()

    env = SnakeEnv(grid_size=args.grid_size)

    if args.random:
        rng = np.random.default_rng(0)
        policy_fn = lambda obs: int(rng.integers(env.action_space.n))
    else:
        if not args.checkpoint:
            parser.error("Provide --checkpoint or --random.")
        agent = PPO.load(args.checkpoint)
        policy_fn = agent.act_greedy

    seeds = list(range(args.best_of))
    frames, score = pick_best_episode(policy_fn, env, seeds, args.max_frames)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps, loop=0)
    print(f"Saved {len(frames)} frames to {args.out} (best score in demo: {score})")


if __name__ == "__main__":
    main()
