import gymnasium as gym
import gymnasium_maze
from gymnasium.utils.play import play

SIZE = 10

mapping = {"d": 0,
           "s": 1,
           "a": 2,
           "w": 3}

play(gym.make("gymnasium_maze/GridWorld-v0", render_mode="rgb_array", size=SIZE),
     keys_to_action=mapping)
