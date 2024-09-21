import time
import pandas as pd
from tqdm import tqdm
from CustomEnv import CustomEnv
from DQNAgent import DQNAgent

def main():
    env = CustomEnv()
    agent = DQNAgent()
    episodes = 1000
    log_data = []

    for episode in tqdm(range(episodes)):
        state = env.reset()
        done = False
        total_reward = 0
        start_time = time.time()

        while not done:
            action = agent.get_action(state.reshape(1, -1))
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        end_time = time.time()
        log_data.append([episode, total_reward, end_time - start_time])

    # Save log data to CSV
    df = pd.DataFrame(log_data, columns=['Episode', 'Total Reward', 'Time Taken'])
    df.to_csv('training_log.csv', index=False)

    # Render the final path
    env.render()

    # Calculate path length
    path_length = len(env.path)
    print(f"Total Path Length: {path_length}")

# Run the main function
if __name__ == "__main__":
    main()
