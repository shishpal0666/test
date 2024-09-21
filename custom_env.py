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
        self.obstacles = [
            (100, 200, 200, 50), 
            (400, 300, 150, 50),
            # Add more obstacles as needed
        ]

    def reset(self):
        self.agent_pos = np.array([100, 100])
        self.path = [self.agent_pos.copy()]  # Track the path taken
        return self.agent_pos

    def step(self, action):
        if action == 0:   # Up
            self.agent_pos[1] += 10
        elif action == 1: # Down
            self.agent_pos[1] -= 10
        elif action == 2: # Left
            self.agent_pos[0] -= 10
        elif action == 3: # Right
            self.agent_pos[0] += 10

        self.agent_pos = np.clip(self.agent_pos, [0, 0], [1000, 1000])

        self.path.append(self.agent_pos.copy())  # Record the position

        # Check if agent has reached the goal
        if np.linalg.norm(self.agent_pos - self.goal_pos) < 10:
            reward = 1
            done = True
        else:
            reward = -0.01
            done = False

        # Check for collisions with obstacles
        for obstacle in self.obstacles:
            if self._check_collision(obstacle):
                reward = -1
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
        plt.plot([0, 1000, 1000, 0, 0], [0, 0, 1000, 1000, 0], color='blue')
        plt.scatter(self.goal_pos[0], self.goal_pos[1], color='red', s=100, label='Goal')
        for obstacle in self.obstacles:
            ox, oy, w, h = obstacle
            plt.gca().add_patch(plt.Rectangle((ox, oy), w, h, color='black'))
        path_array = np.array(self.path)
        plt.plot(path_array[:, 0], path_array[:, 1], color='green', linewidth=2, label='Path Taken')
        plt.legend()
        plt.show()
