import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.replay_memory = []
        self.target_update_counter = 0
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=2, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(4, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(4)  # Explore
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Exploit

    def train(self):
        # Implement training logic based on DQN algorithm here
        pass  # Fill this in later with training logic
