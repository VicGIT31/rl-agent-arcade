"""Custom Snake environment following the Gymnasium API.

Design choices (documented because they matter for training speed and are the
kind of thing a technical interviewer will ask about):

* **Observation is a feature vector, not raw pixels.** An 11-dim vector
  (danger sensors + current direction + relative food direction) lets a small
  MLP learn in minutes on CPU. Pixels would force a CNN and much longer training.
* **Relative action space (3 actions).** The agent chooses to go straight, turn
  right, or turn left. This makes an instant 180-degree reversal (suicide)
  impossible by construction, which stabilises early learning.
* **Reward shaping.** Sparse eat/die rewards plus a small dense signal for
  moving toward the food. See ``_compute_reward`` for the exact scheme.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Absolute directions in clockwise order: UP, RIGHT, DOWN, LEFT.
# Stored as (row_delta, col_delta). Clockwise ordering lets us implement
# "turn right" as (dir + 1) % 4 and "turn left" as (dir - 1) % 4.
_DIRECTIONS = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)], dtype=np.int8)
_UP, _RIGHT, _DOWN, _LEFT = 0, 1, 2, 3

# Relative actions.
ACTION_STRAIGHT = 0
ACTION_RIGHT = 1
ACTION_LEFT = 2


class SnakeEnv(gym.Env):
    """A grid-based Snake game.

    Parameters
    ----------
    grid_size:
        Side length of the square grid.
    max_steps_without_food:
        Episode is truncated if the snake goes this many steps without eating,
        which prevents it from looping forever once it has learned to survive.
    render_mode:
        ``"rgb_array"`` returns an image from :meth:`render`; ``None`` disables it.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(
        self,
        grid_size: int = 12,
        max_steps_without_food: int | None = None,
        render_mode: str | None = "rgb_array",
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.max_steps_without_food = (
            max_steps_without_food
            if max_steps_without_food is not None
            else self.grid_size * self.grid_size
        )
        self.render_mode = render_mode

        # 3 relative actions: straight, turn right, turn left.
        self.action_space = spaces.Discrete(3)
        # 11 binary/normalised features (see _get_obs).
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(11,), dtype=np.float32
        )

        # State initialised in reset().
        self.snake: list[np.ndarray] = []
        self.direction: int = _RIGHT
        self.food: np.ndarray | None = None
        self.steps_since_food: int = 0
        self.score: int = 0

    # ------------------------------------------------------------------ API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        center = self.grid_size // 2
        # Snake of length 3 heading right; head is the first element.
        self.snake = [
            np.array([center, center], dtype=np.int64),
            np.array([center, center - 1], dtype=np.int64),
            np.array([center, center - 2], dtype=np.int64),
        ]
        self.direction = _RIGHT
        self.steps_since_food = 0
        self.score = 0
        self._place_food()
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected 0, 1 or 2.")

        # Resolve relative action into a new absolute direction.
        if action == ACTION_RIGHT:
            self.direction = (self.direction + 1) % 4
        elif action == ACTION_LEFT:
            self.direction = (self.direction - 1) % 4
        # ACTION_STRAIGHT keeps the current direction.

        head = self.snake[0]
        new_head = head + _DIRECTIONS[self.direction]

        prev_dist = self._food_distance(head)
        terminated = self._is_collision(new_head)

        reward = 0.0
        truncated = False

        if terminated:
            reward = -10.0
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        # Move: insert new head.
        self.snake.insert(0, new_head)

        ate = np.array_equal(new_head, self.food)
        if ate:
            reward += 10.0
            self.score += 1
            self.steps_since_food = 0
            self._place_food()
            # Grid full -> the snake has won; end the episode.
            if len(self.snake) >= self.grid_size * self.grid_size:
                terminated = True
        else:
            self.snake.pop()  # move without growing
            self.steps_since_food += 1
            reward += self._compute_reward(prev_dist, new_head)

        if self.steps_since_food >= self.max_steps_without_food:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        """Return an RGB image (H, W, 3) uint8 of the current frame."""
        if self.render_mode != "rgb_array":
            return None
        cell = 20
        size = self.grid_size * cell
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = (15, 15, 20)  # dark background

        def fill(pos, color):
            r, c = int(pos[0]), int(pos[1])
            img[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = color

        # Food.
        if self.food is not None:
            fill(self.food, (220, 60, 60))
        # Body then head (head drawn brighter, on top).
        for segment in self.snake[1:]:
            fill(segment, (60, 180, 90))
        fill(self.snake[0], (120, 240, 150))
        return img

    # -------------------------------------------------------------- helpers

    def _place_food(self) -> None:
        occupied = {tuple(s) for s in self.snake}
        free = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in occupied
        ]
        if not free:  # board full, snake fills the grid
            self.food = None
            return
        idx = self.np_random.integers(len(free))
        self.food = np.array(free[idx], dtype=np.int64)

    def _is_collision(self, point: np.ndarray) -> bool:
        r, c = int(point[0]), int(point[1])
        # Wall.
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return True
        # Body. The tail cell (self.snake[-1]) is about to move away when the
        # snake is not growing, so stepping onto it is safe; exclude it.
        body = self.snake[:-1]
        return any(np.array_equal(point, seg) for seg in body)

    def _food_distance(self, point: np.ndarray) -> float:
        if self.food is None:
            return 0.0
        return float(abs(point[0] - self.food[0]) + abs(point[1] - self.food[1]))

    def _compute_reward(self, prev_dist: float, new_head: np.ndarray) -> float:
        """Dense shaping: nudge the snake toward the food.

        +0.01 when the Manhattan distance to the food decreases, -0.015 when it
        increases. The malus is slightly larger than the bonus so that aimless
        wandering has a small net cost and the agent is pushed to be efficient.
        """
        new_dist = self._food_distance(new_head)
        if new_dist < prev_dist:
            return 0.01
        return -0.015

    def _get_obs(self) -> np.ndarray:
        head = self.snake[0]
        dir_vec = _DIRECTIONS[self.direction]
        right_vec = _DIRECTIONS[(self.direction + 1) % 4]
        left_vec = _DIRECTIONS[(self.direction - 1) % 4]

        obs = np.array(
            [
                # Danger sensors (immediate collision in each relative direction).
                float(self._is_collision(head + dir_vec)),
                float(self._is_collision(head + right_vec)),
                float(self._is_collision(head + left_vec)),
                # Current absolute direction, one-hot.
                float(self.direction == _UP),
                float(self.direction == _RIGHT),
                float(self.direction == _DOWN),
                float(self.direction == _LEFT),
                # Relative food location (booleans).
                float(self.food is not None and self.food[0] < head[0]),  # food up
                float(self.food is not None and self.food[0] > head[0]),  # food down
                float(self.food is not None and self.food[1] < head[1]),  # food left
                float(self.food is not None and self.food[1] > head[1]),  # food right
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> dict:
        return {
            "score": self.score,
            "snake_length": len(self.snake),
            "steps_since_food": self.steps_since_food,
        }
