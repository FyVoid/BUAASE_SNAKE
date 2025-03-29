import torch
from game import *
import random
import numpy as np
from collections import deque

import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

        # Initialize weights using Xavier initialization
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters
GAMMA = 0.95
EPSILON = 1.0
LEARNING_RATE = 0.0001
BATCH_SIZE = 64

# Optimize the model
def optimize_model(policy_net, target_net, optimizer):
    batch = torch.zeros(BATCH_SIZE, 8 * 8 * 7)
    reward = torch.zeros(BATCH_SIZE, 4)
    end = torch.zeros(BATCH_SIZE, 4)
    next_states = torch.zeros(BATCH_SIZE, 4, 8 * 8 * 7)
    for i in range(BATCH_SIZE):
        b, r, e, n = genBoard()
        batch[i, :] = torch.flatten(b)
        reward[i, :] = r
        end[i, :] = e
        next_states[i, :] = n.view(-1, 8 * 8 * 7)

    current_q_values = policy_net(batch)
    action_indices = current_q_values.max(-1)[1]
    if torch.rand(1) < EPSILON:
        action_indices = torch.randint(0, 4, (BATCH_SIZE,))
        current_q_values = current_q_values.gather(1, action_indices.view(-1, 1)).squeeze()
    else:
        current_q_values = current_q_values.max(-1)[0]
    print(action_indices)
    target_states = torch.zeros(BATCH_SIZE, 8 * 8 * 7)
    for i in range(BATCH_SIZE):
        target_states[i, :] = next_states[i, action_indices[i], :]
    target_reward = torch.zeros(BATCH_SIZE)
    for i in range(BATCH_SIZE):
        target_reward[i] = reward[i, action_indices[i]]
    target_end = torch.zeros(BATCH_SIZE)
    for i in range(BATCH_SIZE):
        target_end[i] = end[i, action_indices[i]]
    next_q_values = target_net(target_states).max(1)[0]
    target_q_values = target_reward + (GAMMA * next_q_values * (1 - target_end))

    loss = nn.MSELoss()(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Main training loop
def train_dqn(num_episodes):
    input_dim = 8 * 8 * 7
    output_dim = 4
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    for episode in range(num_episodes):
        global EPSILON
        loss = optimize_model(policy_net, target_net, optimizer)

        target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode}, Total Loss: {loss}")
        
        EPSILON = max(0.1, EPSILON * 0.95)
        print(EPSILON)

    # Save the trained model parameters to a file
    for name, param in policy_net.state_dict().items():
        try:
            np.savetxt(f"{name}.txt", param.cpu().numpy().flatten())  # 转为一行存储
        except Exception:
            pass

    return policy_net

if __name__ == "__main__":
    train_dqn(5000)