"""Interactive demo: watch the trained PPO agent play Snake in the browser.

Runs locally with `python demo/app.py` and deploys as-is to Hugging Face Spaces
(Gradio SDK). The agent plays one greedy episode and the frames are streamed
back as a short animation.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import gradio as gr

# Make the project root importable whether run from repo root or from demo/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.snake_env import SnakeEnv  # noqa: E402
from agents.ppo_scratch import PPO  # noqa: E402

CHECKPOINT = os.environ.get("SNAKE_CHECKPOINT", "checkpoints/best_model.pt")
GRID_SIZE = int(os.environ.get("SNAKE_GRID_SIZE", "12"))

_agent = None


def get_agent():
    global _agent
    if _agent is None and os.path.exists(CHECKPOINT):
        _agent = PPO.load(CHECKPOINT)
    return _agent


def play_episode(seed: int):
    """Play one greedy episode; yield frames so the browser animates the game."""
    env = SnakeEnv(grid_size=GRID_SIZE)
    agent = get_agent()
    obs, _ = env.reset(seed=int(seed))
    frame = env.render()
    done = False
    steps = 0
    score = 0
    if agent is None:
        yield frame, "No checkpoint found — train the agent first (see README)."
        return
    while not done and steps < 600:
        action = agent.act_greedy(obs)
        obs, _, terminated, truncated, info = env.step(action)
        frame = env.render()
        score = info["score"]
        steps += 1
        done = terminated or truncated
        yield frame, f"Score: {score}   |   Steps: {steps}"
    yield frame, f"Episode over — final score: {score} in {steps} steps."


with gr.Blocks(title="RL Snake — PPO from scratch") as demo:
    gr.Markdown(
        "# 🐍 RL Snake — PPO from scratch\n"
        "A Proximal Policy Optimization agent, implemented from scratch, playing "
        "Snake. Pick a seed and press **Play**."
    )
    with gr.Row():
        with gr.Column(scale=1):
            seed = gr.Slider(0, 100, value=0, step=1, label="Episode seed")
            play = gr.Button("▶ Play", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=2):
            screen = gr.Image(label="Game", height=360, width=360)
    play.click(play_episode, inputs=seed, outputs=[screen, status])


if __name__ == "__main__":
    demo.launch()
