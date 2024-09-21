import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 4 possible actions: up, down, left, right
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([1000, 1000]), dtype=np.float32)
        self.agent_pos = np.array([100, 100])
        self.goal_pos = np.array([700, 950])
        self.obstacles = [  # Obstacles as rectangles (x, y, width, height)
            (100, 200, 200, 50), (400, 300, 150, 50), (600, 100, 200, 50), 
            (300, 500, 200, 100), (800, 400, 150, 200), (200, 700, 250, 100),
            (700, 800, 150, 100), (500, 600, 100, 150), (900, 200, 50, 300)
        ]

    def reset(self):
        self.agent_pos = np.array([100, 100])  # Reset to start state
        return self.agent_pos

    def step(self, action):
        if action == 0: self.agent_pos[1] += 10  # Up
        elif action == 1: self.agent_pos[1] -= 10  # Down
        elif action == 2: self.agent_pos[0] -= 10  # Left
        elif action == 3: self.agent_pos[0] += 10  # Right

        self.agent_pos = np.clip(self.agent_pos, [0, 0], [1000, 1000])

        if np.linalg.norm(self.agent_pos - self.goal_pos) < 10:
            reward = 1  # Reached the goal
            done = True
        else:
            reward = -0.01  # Time penalty
            done = False

        for obstacle in self.obstacles:
            if self._check_collision(obstacle):
                reward = -1  # Collision
                done = True
                break

        return self.agent_pos, reward, done, {}

    def _check_collision(self, obstacle):
        ox, oy, w, h = obstacle
        return (ox <= self.agent_pos[0] <= ox + w) and (oy <= self.agent_pos[1] <= oy + h)

    def render(self, mode='human'):
        plt.figure(figsize=(10, 10))
        plt.xlim(0, 1000)
        plt.ylim(0, 1000)
        plt.scatter(self.agent_pos[0], self.agent_pos[1], color='green', s=100, label='Agent')
        plt.scatter(self.goal_pos[0], self.goal_pos[1], color='red', s=100, label='Goal')
        for obstacle in self.obstacles:
            ox, oy, w, h = obstacle
            plt.gca().add_patch(plt.Rectangle((ox, oy), w, h, color='black'))
        plt.legend()
        plt.show()
