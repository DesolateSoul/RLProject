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
        self.vision_radius = 2  # Радиус обзора агента

        # Пространство наблюдений теперь содержит только ближайшие препятствия (максимум 8 в радиусе 2)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=np.int64),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=np.int64),
                "visible_holes": spaces.Box(0, size - 1, shape=(8, 2), dtype=np.int64)  # Макс 8 дыр в радиусе 2
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
        """Генерация дыр с проверкой доступности старта и финиша"""
        for _ in range(100):  # Максимум 100 попыток генерации
            self._holes = []

            # 1. Добавляем дыры по краям
            for i in range(self.size):
                self._holes.append([i, 0])  # Нижний край
                self._holes.append([i, self.size - 1])  # Верхний край
                self._holes.append([0, i])  # Левый край
                self._holes.append([self.size - 1, i])  # Правый край

            # 2. Добавляем случайные дыры внутри
            for _ in range((self.size - 2) ** 2 // 4):
                x, y = self.np_random.integers(1, self.size - 1, size=2)
                while [x, y] in self._holes:
                    x, y = self.np_random.integers(1, self.size - 1, size=2)
                if [x, y] not in self._holes:
                    self._holes.append([x, y])

            # Проверяем доступность старта и финиша
            start_pos = np.array([1, 1])
            target_pos = np.array([self.size - 2, self.size - 2])

            if (not any(np.array_equal(start_pos, hole) for hole in self._holes) and
                    not any(np.array_equal(target_pos, hole) for hole in self._holes) and
                    self._is_position_valid(start_pos) and
                    self._is_position_valid(target_pos)):
                return np.array(self._holes, dtype=np.int64)

        raise RuntimeError("Не удалось создать валидную карту после 100 попыток")

    def _is_position_valid(self, pos):
        """Проверяет, что позиция не окружена дырами со всех сторон"""
        x, y = pos
        directions = [np.array([1, 0]), np.array([-1, 0]),
                      np.array([0, 1]), np.array([0, -1])]

        # Проверяем наличие хотя бы одного свободного соседа
        for direction in directions:
            neighbor = pos + direction
            if (0 <= neighbor[0] < self.size and
                    0 <= neighbor[1] < self.size and
                    not any(np.array_equal(neighbor, hole) for hole in self._holes)):
                return True
        return False

    def _get_visible_holes(self, agent_pos):
        """Возвращает только дыры в радиусе обзора агента"""
        visible = []
        for hole in self._holes:
            if np.linalg.norm(agent_pos - hole, ord=1) <= self.vision_radius:
                visible.append(hole)

        # Дополняем нулями до 8 дыр (для фиксированного размера)
        while len(visible) < 8:
            visible.append([0, 0])  # Пустые позиции

        return np.array(visible[:8], dtype=np.int64)  # Возвращаем не более 8 дыр

    def _get_obs(self):
        visible_holes = self._get_visible_holes(self._agent_location)
        return {
            "agent": np.array(self._agent_location, dtype=np.int64),
            "target": np.array(self._target_location, dtype=np.int64),
            "visible_holes": visible_holes
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1),
            "near_hole": any(np.array_equal(self._agent_location, hole) for hole in self._holes)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Агент в позиции (1, 1)
        self._agent_location = np.array([1, 1], dtype=np.int64)

        # Цель в позиции (size-2, size-2)
        self._target_location = np.array([self.size - 2, self.size - 2], dtype=np.int64)

        # Генерация препятствий с проверками
        self._generate_holes()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Проверка на попадание в яму
        in_hole = any(np.array_equal(self._agent_location, hole) for hole in self._holes)
        terminated = np.array_equal(self._agent_location, self._target_location)

        # Награда: +50 за достижение цели, -10 за попадание в яму, -1 за каждый шаг
        reward = 50 if terminated else (-10 if in_hole else -1)

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

        # Рисуем дыры (черные квадраты)
        for hole in self._holes:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    (pix_square_size * hole[0], pix_square_size * hole[1]),  # Разделяем x и y
                    (pix_square_size, pix_square_size),
                ),
            )

        # Остальной код рендеринга (цель, агент, сетка)
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                (pix_square_size * self._target_location[0], pix_square_size * self._target_location[1]),
                (pix_square_size, pix_square_size),
            ),
        )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            ((self._agent_location[0] + 0.5) * pix_square_size,
             (self._agent_location[1] + 0.5) * pix_square_size),
            pix_square_size / 3,
        )

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
