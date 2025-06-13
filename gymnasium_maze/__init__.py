from gymnasium.envs.registration import register

register(
    id="gymnasium_maze/GridWorld-v0",
    entry_point="gymnasium_maze.envs:GridWorldEnv",
)
