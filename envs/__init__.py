"""Custom Gymnasium environments."""

from gymnasium.envs.registration import register

from envs.snake_env import SnakeEnv

register(
    id="Snake-v0",
    entry_point="envs.snake_env:SnakeEnv",
    max_episode_steps=None,
)

__all__ = ["SnakeEnv"]
