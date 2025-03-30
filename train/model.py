import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class DQN(nn.Module):
    def __init__(self, state_features=config.STATE_FEATURES, state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE,
                 hidden1=config.HIDDEN_LAYER_1, hidden2=config.HIDDEN_LAYER_2, hidden3 = config.HIDDEN_LAYER_3):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=state_features, out_channels=hidden1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden1, out_channels=hidden2, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=hidden2, out_channels=hidden3, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(hidden3 * int(state_size / state_features), action_size)

    def forward(self, state):
        state = state.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor for the fully connected layer
        x = self.fc(x)
        return x
