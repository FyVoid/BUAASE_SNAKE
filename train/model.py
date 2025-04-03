import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # Compute attention scores
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        weights = F.softmax(scores, dim=-1)
        
        # Compute weighted sum of values
        output = torch.matmul(weights, value)
        return output, weights

class DQN(nn.Module):
    def __init__(self, state_features=config.STATE_FEATURES, state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE,
        hidden1=config.HIDDEN_LAYER_1, hidden2=config.HIDDEN_LAYER_2, hidden3 = config.HIDDEN_LAYER_3):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(in_channels=state_features, out_channels=hidden1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden1, out_channels=hidden2, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(in_channels=hidden2, out_channels=hidden1, kernel_size=3, stride=1, padding=1)
        conv1_out_size = state_size  # conv1 keeps size same due to padding=1 (kernel_size=3, stride=1)
        conv2_out_size = conv1_out_size  # conv2 keeps size same due to padding=2 (kernel_size=5, stride=1)
        conv3_out_size = conv2_out_size  # conv3 keeps size same due to padding=3 (kernel_size=7, stride=1)
        conv4_out_size = conv3_out_size  # conv4 keeps size same due to padding=2 (kernel_size=5, stride=1)
        conv5_out_size = conv4_out_size  # conv5 keeps size same due to padding=1 (kernel_size=3, stride=1)
        self.out = nn.Linear(hidden1 * conv5_out_size * conv5_out_size, action_size)

    def forward(self, state):
        state = state.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv5(x))
        x = self.flatten(x)
        x = self.out(x)
        return x
