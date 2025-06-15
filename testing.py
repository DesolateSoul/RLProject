import numpy as np
from collections import defaultdict
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple
import pickle
import gymnasium as gym
import gymnasium_maze
from agent import Agent


def process_state(observation):
    """Нормализует наблюдение для нейросетевого агента"""
    agent_pos = np.array(observation['agent'], dtype=np.float32) / 9.0
    target_pos = np.array(observation['target'], dtype=np.float32) / 9.0
    holes_pos = np.array(observation['visible_holes'], dtype=np.float32).flatten() / 9.0
    return np.concatenate([agent_pos, target_pos, holes_pos])


class Network(nn.Module):

    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)


maximum_number_timesteps_per_episode = 100


def test_agent(env, agent, episodes=100):
    """Тестирует агента в среде и собирает статистику"""
    stats = {
        'success_count': 0,
        'total_holes': 0,
        'total_steps': 0,
        'holes_per_episode': []
    }

    for episode in range(episodes):
        obs, _ = env.reset()
        processed_obs = process_state(obs)
        done = False
        holes_passed = 0
        reward = 0
        terminated = False

        for t in range(maximum_number_timesteps_per_episode):
            # Получаем действие от агента на основе обработанного состояния
            action = agent.act(processed_obs)

            # Выполняем шаг в среде
            next_obs, reward, terminated, truncated, info = env.step(action)
            processed_next_obs = process_state(next_obs)
            done = terminated or truncated

            # Подсчет пройденных дыр
            if reward == -10:
                holes_passed += 1

            processed_obs = processed_next_obs
            stats['total_steps'] += 1

            if done:
                break

        # Проверяем успешное завершение
        if terminated and reward == 50:
            stats['success_count'] += 1

        stats['total_holes'] += holes_passed
        stats['holes_per_episode'].append(holes_passed)

    # Рассчитываем метрики
    success_rate = stats['success_count'] / episodes
    avg_holes = stats['total_holes'] / episodes
    avg_steps = stats['total_steps'] / episodes

    print("\nРезультаты тестирования:")
    print(f"Эпизодов: {episodes}")
    print(f"Успешных достижений цели: {stats['success_count']} ({success_rate:.1%})")
    print(f"Среднее количество пройденных препятствий за эпизод: {avg_holes:.2f}")
    print(f"Среднее количество шагов: {avg_steps:.2f}")

    return {
        'success_rate': success_rate,
        'avg_holes_passed': avg_holes,
        'avg_steps': avg_steps,
        'holes_distribution': stats['holes_per_episode']
    }


# Создаем среду и агента
env = gym.make('gymnasium_maze/GridWorld-v0', size=12)
with open("agent.pkl", "rb") as fp:
    agent = pickle.load(fp)


# Запускаем тестирование
test_results = test_agent(env, agent, episodes=100)
