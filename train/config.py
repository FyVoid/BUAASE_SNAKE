# config.py
import torch

GRID_SIZE = 8       # Should match snake_game.py if not passed explicitly
STATE_FEATURES = 7  # Number of features per cell (one-hot encoding length)

BUFFER_SIZE = 512      # Replay memory size
BATCH_SIZE = 128         # Number of experiences to sample for learning
GAMMA = 0.95             # Discount factor for future rewards
EPS_START = 1.0          # Starting value of epsilon (exploration rate)
EPS_END = 0.05           # Minimum value of epsilon
EPS_DECAY = 1000         # Controls the rate of exponential decay of epsilon
LR = 1e-4                # Learning rate for the optimizer
TARGET_UPDATE_FREQ = 128  # How often to update the target network (in steps or episodes) - using steps here

# Simple MLP: Flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear (Output)
HIDDEN_LAYER_1 = 32
HIDDEN_LAYER_2 = 64
HIDDEN_LAYER_3 = 32

NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 200 # Prevent infinitely running episodes
LOG_INTERVAL = 50           # How often to print training progress
SAVE_INTERVAL = 1000         # How often to save the model
MODEL_SAVE_PATH = "dqn_snake_model.pth"
PARAMS_SAVE_PATH = "dqn_snake_params.bin" # Path for simplified parameter saving

DEVICE = "cuda" if torch.cuda.is_available else "cpu"
print(f"Using device: {DEVICE}")

STATE_SIZE = GRID_SIZE * GRID_SIZE * STATE_FEATURES
ACTION_SIZE = 4