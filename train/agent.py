from collections import namedtuple, deque
import random as rd
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import config
from model import DQN

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward', 'done')
)

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        if len(self.memory) < batch_size:
            raise ValueError("Not enough samples in memory to sample the batch.")
        return rd.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        return len(self.memory)
    
class DQNAgent:

    def __init__(self, state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = rd.seed(seed)

        rd.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.policy_net = DQN(state_size, action_size).to(config.DEVICE)
        self.target_net = DQN(state_size, action_size).to(config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LR)

        self.memory = deque(maxlen=config.BUFFER_SIZE)

        self.epsilon = config.EPS_START
        self.eps_decay_value = (config.EPS_START - config.EPS_END) / config.EPS_DECAY
        self.steps_done = 0

    def step(self, state, action, reward, next_state, done):
        transition = Transition(state, action, next_state, reward, done)
        self.memory.append(transition)
        self.steps_done += 1

        if len(self.memory) > config.BATCH_SIZE:
            transitions = self._sample(config.BATCH_SIZE)
            self._learn(transitions, config.GAMMA)

        if self.steps_done % config.TARGET_UPDATE_FREQ == 0:
             self._update_target_network()

    def select_action(self, state, evaluate=False):
        if state.dim() == 3:
             state = state.unsqueeze(0)

        state = state.to(config.DEVICE)

        effective_eps = config.EPS_END if evaluate else self.epsilon
        if rd.random() > effective_eps:
            self.policy_net.eval()
            with torch.no_grad():
                action_values = self.policy_net(state)
            self.policy_net.train()
            action = np.argmax(action_values.cpu().data.numpy())
            return action
        else:
            return rd.choice(np.arange(self.action_size))
        
    def _learn(self, transitions, gamma):
        states, actions, rewards, next_states, dones = transitions
        with torch.no_grad():
            Q_targets_next = self.target_net(next_states).max(1)[0].unsqueeze(1) # Shape: (batch_size, 1)

        # Compute Q targets for current states: R + gamma * max_a' Q_target(s', a')
        # If the state is terminal (done=True), the target is just the reward
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) # Shape: (batch_size, 1)

        # Get expected Q values from policy model for the actions actually taken
        # action tensor needs to be shape (batch_size, 1) and Long dtype for gather
        actions = actions.long()
        Q_expected = self.policy_net(states).gather(1, actions) # Shape: (batch_size, 1)

        # Compute loss (Mean Squared Error)
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Clip gradients to prevent exploding gradients
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(config.EPS_END, self.epsilon - self.eps_decay_value)


    def _update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _sample(self, batch_size):
        transitions = rd.sample(self.memory, k=config.BATCH_SIZE)

        # Convert batch of Experiences to tensors on the correct device
        states = torch.stack([e.state for e in transitions if e is not None]).float().to(config.DEVICE)
        # Ensure actions are column vectors (N, 1) instead of (N,)
        actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long().to(config.DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float().to(config.DEVICE)
        next_states = torch.stack([e.next_state for e in transitions if e is not None]).float().to(config.DEVICE)
        # Convert dones (boolean) to float (0.0 or 1.0) for calculations
        dones = torch.from_numpy(np.vstack([e.done for e in transitions if e is not None]).astype(np.uint8)).float().to(config.DEVICE)

        return (states, actions, rewards, next_states, dones)

    def save_model(self, path=config.MODEL_SAVE_PATH):
        """Saves the policy network's state dictionary."""
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path=config.MODEL_SAVE_PATH):
        """Loads the policy network's state dictionary."""
        self.policy_net.load_state_dict(torch.load(path, map_location=config.DEVICE))
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Sync target net
        self.policy_net.to(config.DEVICE)
        self.target_net.to(config.DEVICE)
        self.target_net.eval()
        print(f"Model loaded from {path}")