"""Unit tests for the Snake environment.

These validate the Gymnasium contract (reset/step shapes, spaces) and the
game-specific invariants (rewards, collisions, food, growth). The rule from the
project plan: PPO is not touched until these pass and a random agent runs.
"""

import numpy as np
import pytest

from envs.snake_env import (
    SnakeEnv,
    ACTION_STRAIGHT,
    ACTION_RIGHT,
    ACTION_LEFT,
)


@pytest.fixture
def env():
    e = SnakeEnv(grid_size=12)
    e.reset(seed=0)
    return e


def test_reset_returns_valid_observation(env):
    obs, info = env.reset(seed=123)
    assert env.observation_space.contains(obs)
    assert obs.shape == (11,)
    assert obs.dtype == np.float32
    assert info["score"] == 0
    assert info["snake_length"] == 3


def test_reset_is_deterministic_with_seed():
    a = SnakeEnv(grid_size=12)
    b = SnakeEnv(grid_size=12)
    obs_a, _ = a.reset(seed=42)
    obs_b, _ = b.reset(seed=42)
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(a.food, b.food)


def test_step_returns_five_tuple_and_valid_obs(env):
    result = env.step(ACTION_STRAIGHT)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_invalid_action_raises(env):
    with pytest.raises(ValueError):
        env.step(7)


def test_wall_collision_terminates_with_penalty():
    env = SnakeEnv(grid_size=6)
    env.reset(seed=0)
    # Snake starts at center heading RIGHT. Drive straight into the right wall.
    terminated = False
    reward = 0.0
    for _ in range(env.grid_size):
        _, reward, terminated, _, _ = env.step(ACTION_STRAIGHT)
        if terminated:
            break
    assert terminated is True
    assert reward == -10.0


def test_eating_food_increases_score_and_length():
    env = SnakeEnv(grid_size=8)
    env.reset(seed=0)
    # Force the food directly in front of the head so a straight step eats it.
    head = env.snake[0].copy()
    env.food = head + np.array([0, 1])  # one cell to the right (heading right)
    length_before = len(env.snake)
    _, reward, terminated, _, info = env.step(ACTION_STRAIGHT)
    assert reward == pytest.approx(10.0)
    assert terminated is False
    assert info["score"] == 1
    assert len(env.snake) == length_before + 1


def test_moving_without_eating_keeps_length():
    env = SnakeEnv(grid_size=10)
    env.reset(seed=1)
    # Put the food far away so a single step cannot eat it.
    env.food = np.array([0, 0], dtype=np.int64)
    length_before = len(env.snake)
    env.step(ACTION_STRAIGHT)
    assert len(env.snake) == length_before


def test_shaping_reward_sign():
    env = SnakeEnv(grid_size=10)
    env.reset(seed=2)
    # Snake heads RIGHT from center; place food to the right -> moving straight
    # decreases distance -> positive shaping reward.
    head = env.snake[0].copy()
    env.food = np.array([head[0], env.grid_size - 1], dtype=np.int64)
    _, reward, _, _, _ = env.step(ACTION_STRAIGHT)
    assert reward > 0

    # Now place food to the left of a rightward-heading snake -> moving straight
    # increases distance -> negative shaping reward.
    env.reset(seed=2)
    head = env.snake[0].copy()
    env.food = np.array([head[0], 0], dtype=np.int64)
    _, reward, _, _, _ = env.step(ACTION_STRAIGHT)
    assert reward < 0


def test_relative_turns_change_direction():
    env = SnakeEnv(grid_size=12)
    env.reset(seed=0)
    d0 = env.direction
    env.step(ACTION_RIGHT)
    assert env.direction == (d0 + 1) % 4
    env.step(ACTION_LEFT)
    assert env.direction == d0  # turned back


def test_food_never_spawns_on_snake():
    env = SnakeEnv(grid_size=6)
    env.reset(seed=0)
    for _ in range(200):
        occupied = {tuple(s) for s in env.snake}
        if env.food is not None:
            assert tuple(env.food) not in occupied
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()


def test_render_returns_rgb_image(env):
    img = env.render()
    assert img.shape == (env.grid_size * 20, env.grid_size * 20, 3)
    assert img.dtype == np.uint8


def test_random_agent_runs_full_episode():
    """A random agent should be able to play start-to-finish without crashing."""
    env = SnakeEnv(grid_size=10)
    env.reset(seed=0)
    for _ in range(1000):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
            assert env.observation_space.contains(obs)
