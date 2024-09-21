import gym
import torch
import numpy as np
import pandas as pd
from dqn_model import DQN, ReplayBuffer
from custom_env import CustomEnv

env = CustomEnv()
num_actions = env.action_space.n
model = DQN(num_actions)
optimizer = torch.optim.Adam(model.parameters())
replay_buffer = ReplayBuffer(10000)
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
num_episodes = 1000

# CSV file to log results
log_data = []

def train_model():
    global epsilon
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        total_reward = 0
        done = False
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = model(state).max(1)[1].item()
            
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if replay_buffer.size() > batch_size:
                train_step()

        # Log episode metrics
        log_data.append([episode, total_reward])

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    # Save CSV
    df = pd.DataFrame(log_data, columns=["Episode", "Reward"])
    df.to_csv('results/training_log.csv', index=False)

def train_step():
    states, actions, rewards, next_states, dones = zip(*replay_buffer.sample(batch_size))
    states = torch.cat(states)
    next_states = torch.cat(next_states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    q_values = model(states).gather(1, actions)
    next_q_values = model(next_states).max(1)[0].unsqueeze(1)
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    loss = (q_values - target_q_values.detach()).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

train_model()
