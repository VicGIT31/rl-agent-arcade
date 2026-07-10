"""Training loop for PPO-from-scratch on the Snake environment.

Usage:
    python -m training.train --config training/configs/snake_ppo.yaml

Logs episode reward, episode length, game score, and PPO losses to TensorBoard,
and periodically saves checkpoints (used later to build the progression GIFs).
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
import yaml

from envs.snake_env import SnakeEnv
from agents.ppo_scratch import PPO, PPOConfig


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(agent: PPO, env: SnakeEnv, episodes: int, seed: int = 10_000) -> dict:
    """Greedy evaluation over several episodes. Returns mean reward/score/length."""
    rewards, scores, lengths = [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        total_r, steps = 0.0, 0
        while not done:
            action = agent.act_greedy(obs)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            steps += 1
            done = terminated or truncated
        rewards.append(total_r)
        scores.append(info["score"])
        lengths.append(steps)
    return {
        "eval/mean_reward": float(np.mean(rewards)),
        "eval/mean_score": float(np.mean(scores)),
        "eval/max_score": float(np.max(scores)),
        "eval/mean_length": float(np.mean(lengths)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/configs/snake_ppo.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override total_timesteps from the config.")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg, ppo_cfg, train_cfg = cfg["env"], cfg["ppo"], cfg["train"]
    if args.total_timesteps is not None:
        train_cfg["total_timesteps"] = args.total_timesteps

    run_name = args.run_name or f"snake_ppo_{int(time.time())}"
    log_dir = os.path.join(train_cfg["log_dir"], run_name)
    ckpt_dir = train_cfg["checkpoint_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # TensorBoard is optional at import time so the trainer still runs without it.
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] TensorBoard unavailable ({exc}); logging to stdout only.")

    env = SnakeEnv(
        grid_size=env_cfg["grid_size"],
        max_steps_without_food=env_cfg["max_steps_without_food"],
    )
    eval_env = SnakeEnv(
        grid_size=env_cfg["grid_size"],
        max_steps_without_food=env_cfg["max_steps_without_food"],
    )

    ppo_config = PPOConfig(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        seed=train_cfg["seed"],
        device=train_cfg["device"],
        **ppo_cfg,
    )
    agent = PPO(ppo_config)

    total_timesteps = train_cfg["total_timesteps"]
    rollout_steps = ppo_config.rollout_steps
    n_updates = total_timesteps // rollout_steps

    obs, _ = env.reset(seed=train_cfg["seed"])
    ep_reward, ep_len = 0.0, 0
    reward_window = deque(maxlen=100)
    score_window = deque(maxlen=100)
    global_step = 0
    best_score = -np.inf
    start = time.time()

    print(f"Training for {n_updates} updates "
          f"({total_timesteps:,} timesteps, rollout={rollout_steps}).")

    for update in range(1, n_updates + 1):
        # ---- collect one on-policy rollout ----
        for _ in range(rollout_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.buffer.add(obs, action, log_prob, reward, float(done), value)
            obs = next_obs
            ep_reward += reward
            ep_len += 1
            global_step += 1
            if done:
                reward_window.append(ep_reward)
                score_window.append(info["score"])
                obs, _ = env.reset()
                ep_reward, ep_len = 0.0, 0

        # Bootstrap value of the state we stopped on (0 if it was terminal).
        last_value = 0.0 if done else agent.get_value(obs)
        stats = agent.update(last_value)

        # ---- logging ----
        mean_r = float(np.mean(reward_window)) if reward_window else 0.0
        mean_s = float(np.mean(score_window)) if score_window else 0.0
        if writer:
            writer.add_scalar("rollout/mean_ep_reward", mean_r, global_step)
            writer.add_scalar("rollout/mean_score", mean_s, global_step)
            for k, v in stats.items():
                writer.add_scalar(f"loss/{k}", v, global_step)

        if update % 10 == 0 or update == 1:
            elapsed = time.time() - start
            fps = int(global_step / max(elapsed, 1e-6))
            print(f"upd {update:>4}/{n_updates} | step {global_step:>8,} | "
                  f"reward100 {mean_r:6.2f} | score100 {mean_s:5.2f} | "
                  f"entropy {stats['entropy']:.3f} | {fps} fps")

        # ---- periodic evaluation ----
        if update % train_cfg["eval_every_updates"] == 0:
            eval_stats = evaluate(agent, eval_env, train_cfg["eval_episodes"])
            if writer:
                for k, v in eval_stats.items():
                    writer.add_scalar(k, v, global_step)
            if eval_stats["eval/mean_score"] > best_score:
                best_score = eval_stats["eval/mean_score"]
                agent.save(os.path.join(ckpt_dir, "best_model.pt"))

        # ---- periodic checkpoints (for progression GIFs) ----
        if update % train_cfg["checkpoint_every_updates"] == 0 or update == 1:
            agent.save(os.path.join(ckpt_dir, f"ckpt_update_{update:05d}.pt"))

    agent.save(os.path.join(ckpt_dir, "final_model.pt"))
    if writer:
        writer.close()
    print(f"Done in {time.time() - start:.1f}s. Best eval score: {best_score:.2f}")


if __name__ == "__main__":
    main()
