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


env = gym.make('gymnasium_maze/GridWorld-v0')


def process_state(observation):
    # Преобразуем все элементы в numpy arrays, если они ещё не являются таковыми
    agent_pos = np.array(observation['agent'], dtype=np.float32) / 9.0
    target_pos = np.array(observation['target'], dtype=np.float32) / 9.0
    holes_pos = np.array(observation['holes'], dtype=np.float32).flatten() / 9.0
    return np.concatenate([agent_pos, target_pos, holes_pos])


# Для GridWorld нужно обработать словарное observation space
# Мы будем объединять координаты агента и цели в один плоский вектор
state_size = 20  # agent_x, agent_y, target_x, target_y
number_actions = env.action_space.n
print('State size: ', state_size)
print('Number of actions: ', number_actions)
# Initializing the DQN agent

agent = Agent(state_size, number_actions)

# Training the DQN agent

number_episodes = 2000
maximum_number_timesteps_per_episode = 1000
epsilon_starting_value = 1.0
epsilon_ending_value = 0.01
epsilon_decay_value = 0.995
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen=100)
all_scores = []

for episode in range(1, number_episodes + 1):
    observation, _ = env.reset()
    state = process_state(observation)  # Преобразование наблюдения в плоский вектор
    score = 0
    for t in range(maximum_number_timesteps_per_episode):
        action = agent.act(state, epsilon)
        next_observation, reward, done, _, _ = env.step(action)
        next_state = process_state(next_observation)  # Преобразование следующего наблюдения
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    scores_on_100_episodes.append(score)
    all_scores.append(score)
    epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end="")
    if episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
    if np.mean(scores_on_100_episodes) >= 98 and episode % 100 == 0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode,
                                                                                     np.mean(scores_on_100_episodes)))
        with open("agent.pkl", "wb") as fp:
            pickle.dump(agent, fp)
        break
    if episode == number_episodes:
        with open("agent.pkl", "wb") as fp:
            pickle.dump(agent, fp)
        break
