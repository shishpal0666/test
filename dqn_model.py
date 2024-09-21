import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.cnn_layer = deepmind()  # From the first code you provided
        self.fc1 = nn.Linear(32 * 7 * 7, 256)
        self.action_value = nn.Linear(256, num_actions)

    def forward(self, inputs):
        x = self.cnn_layer(inputs)
        x = F.relu(self.fc1(x))
        return self.action_value(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)
