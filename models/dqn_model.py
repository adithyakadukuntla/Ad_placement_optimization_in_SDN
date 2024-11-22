import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_dqn(env, episodes=1000):
    state_size = env.state_space.shape[0]
    action_size = env.action_space.n
    model = DQN(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995
    gamma = 0.99  # Discount factor
    memory = []

    batch_size = 64
    total_rewards = []  # Store rewards for each episode
    actions_taken = []  # Track actions taken during each episode

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_actions = []  # Actions taken in the current episode

        for t in range(200):  # Limit steps per episode
            state_tensor = torch.FloatTensor(state)
            if np.random.rand() < epsilon:
                action = np.random.choice(action_size)  # Explore
            else:
                with torch.no_grad():
                    action = torch.argmax(model(state_tensor)).item()  # Exploit

            next_state, reward, done, _ = env.step(action)

            # Append transition to memory
            memory.append((state, action, reward, next_state, done))

            # Ensure memory doesn't exceed a maximum size
            if len(memory) > 10000:
                memory.pop(0)

            total_reward += reward
            state = next_state
            episode_actions.append(action)  # Track action

            # Train the model if memory is sufficient
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Convert to tensors
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                # Q-value updates
                q_values = model(states)
                next_q_values = model(next_states).detach()
                target_q_values = rewards + gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones)

                # Gather Q-values for taken actions
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute loss and optimize
                loss = loss_fn(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        epsilon = max(0.1, epsilon * epsilon_decay)  # Decay epsilon
        total_rewards.append(total_reward)  # Append episode reward
        actions_taken.append(episode_actions)  # Store actions taken in the episode
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

    return model, total_rewards, actions_taken

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance")
    plt.legend()
    plt.show()

def calculate_action_rewards(actions_taken, total_rewards, action_size):
    # Calculate average reward for each action (ad position)
    action_rewards = {i: [] for i in range(action_size)}
    
    for episode, actions in enumerate(actions_taken):
        for i, action in enumerate(actions):
            action_rewards[action].append(total_rewards[episode])  # Store reward for each action taken
    
    # Average rewards per action
    average_rewards = {action: np.mean(rewards) for action, rewards in action_rewards.items()}
    print("Average rewards per action:", average_rewards)


# Example usage:
# Assuming `env` is your environment and `total_rewards` is a list storing rewards from each episode
# env = None  # Replace this with your actual environment
# model, total_rewards, actions_taken = train_dqn(env)

# Plot rewards
# plot_rewards(total_rewards)

# # Calculate and display average rewards for each action (ad position)
# calculate_action_rewards(actions_taken, total_rewards, env.action_space.n)
