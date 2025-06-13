from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10):
        self.size = size
        self.window_size = 512

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=np.int64),  # Явно укажите int64
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=np.int64),
                "holes": spaces.Box(0, size - 1, shape=(size - 2, 2), dtype=np.int64)
            }
        )

        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._holes = None

    def _generate_holes(self):
        """Генерирует по одной яме в каждом ряду, исключая старт и финиш"""
        self._holes = []
        for y in range(self.size):
            if y == 0 or y == self.size - 1:  # Пропускаем ряды с агентом и целью
                continue
            x = self.np_random.integers(0, self.size - 1)
            while (x == 0 and y == 0) or (x == self.size - 1 and y == self.size - 1):
                x = self.np_random.integers(0, self.size)
            self._holes.append([x, y])
        # Преобразуем список в numpy array
        return np.array(self._holes, dtype=np.int64)

    def _get_obs(self):
        return {
            "agent": np.array(self._agent_location, dtype=np.int64),
            "target": np.array(self._target_location, dtype=np.int64),
            "holes": np.array(self._holes, dtype=np.int64)  # Гарантируем numpy array
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1),
            "near_hole": any(np.array_equal(self._agent_location, hole) for hole in self._holes)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = np.array([0, 0], dtype=np.int64)  # Левая нижняя клетка
        self._target_location = np.array([self.size - 1, self.size - 1], dtype=np.int64)  # Правая верхняя
        self._holes = self._generate_holes()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Проверка на попадание в яму
        in_hole = any(np.array_equal(self._agent_location, hole) for hole in self._holes)
        terminated = np.array_equal(self._agent_location, self._target_location)

        # Награда: +100 за достижение цели, -1 за попадание в яму, -0.1 за каждый шаг
        reward = 100 if terminated else (-100 if in_hole else -0.1)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # Рисуем ямы (черные квадраты)
        for hole in self._holes:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * hole,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Рисуем цель (красный квадрат)
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Рисуем агента (синий круг)
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Рисуем сетку
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
