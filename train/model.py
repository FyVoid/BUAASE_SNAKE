import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class DQN(nn.Module):
    def __init__(self, state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE,
                 hidden1=config.HIDDEN_LAYER_1, hidden2=config.HIDDEN_LAYER_2):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)

    def forward(self, state):
        x = self.flatten(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
