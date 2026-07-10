"""Plot the learning curve from a TensorBoard run into a PNG for the README.

Reads the scalar events written during training and plots mean episode reward
and mean game score against environment steps.

Usage:
    python -m evaluation.plot_curves --logdir runs/snake_ppo_main --out media/learning_curve.png
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_scalars(logdir: str, tag: str):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = glob.glob(os.path.join(logdir, "**", "events.*"), recursive=True)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files under {logdir}")
    acc = EventAccumulator(logdir)
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        raise KeyError(f"Tag {tag!r} not found. Available: {acc.Tags()['scalars']}")
    events = acc.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="runs/snake_ppo_main")
    parser.add_argument("--out", default="media/learning_curve.png")
    args = parser.parse_args()

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    steps_r, reward = load_scalars(args.logdir, "rollout/mean_ep_reward")
    ax1.plot(steps_r, reward, color="#2b8a3e", label="mean episode reward")
    ax1.set_xlabel("environment steps")
    ax1.set_ylabel("mean episode reward", color="#2b8a3e")
    ax1.tick_params(axis="y", labelcolor="#2b8a3e")
    ax1.grid(alpha=0.3)

    try:
        steps_s, score = load_scalars(args.logdir, "rollout/mean_score")
        ax2 = ax1.twinx()
        ax2.plot(steps_s, score, color="#1c7ed6", label="mean game score")
        ax2.set_ylabel("mean game score (food eaten)", color="#1c7ed6")
        ax2.tick_params(axis="y", labelcolor="#1c7ed6")
    except KeyError:
        pass

    plt.title("PPO-from-scratch learning curve — Snake")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
