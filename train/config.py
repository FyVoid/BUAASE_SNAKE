# config.py
import torch

SEED = 617

STATE_FEATURES = 8  # Number of features per cell (one-hot encoding length)

MODEL_PATH = None

# GAME_MODE = "1v1"
GAME_MODE = "4p"

GRID_SIZE = 5 if GAME_MODE == "1v1" else 8

BUFFER_SIZE = 25600      # Replay memory size
BATCH_SIZE = 256         # Number of experiences to sample for learning
GAMMA = 0.8             # Discount factor for future rewards
EPS_START = 1.0          # Starting value of epsilon (exploration rate)
EPS_END = 0.05           # Minimum value of epsilon
EPS_DECAY = 8000         # Controls the rate of exponential decay of epsilon
LR = 3e-4                # Learning rate for the optimizer
TARGET_UPDATE_FREQ = 4000  # How often to update the target network (in steps or episodes) - using steps here

HIDDEN_LAYER_1 = 32
HIDDEN_LAYER_2 = 64

NUM_FOODS = 5 if GAME_MODE == "1v1" else 10
ENEMY_SNAKE_COUNT = 1 if GAME_MODE == "1v1" else 3
MAX_STEP_PER_GAME = 50 if GAME_MODE == "1v1" else 100

REWARD = {
    "DEATH_EARLY": -30.0,
    "SELF_KILL": -25.0,
    "DEATH_LATE": -20.0,
    "LIVING": 1.0,
    
    "FOOD": 8.0,
    "FOOD_ON_ENEMY_CORPSE": 16.0,
    "CLOSE_TO_FOOD": 3.0,
    
    "KILL": 20.0
}

# max_reward = max(REWARD.values())
# min_reward = min(REWARD.values())
# # Normalize rewards to be between 0 and 1
# for key in REWARD.keys():
#     REWARD[key] = (REWARD[key] - min_reward) / (max_reward - min_reward)

SEARCH_DISTANCE = 1 if GAME_MODE == "1v1" else 2

INTERACT = False
NUM_EPISODES = 20000
MAX_STEPS_PER_EPISODE = 200 # Prevent infinitely running episodes
EVAL_START_EPISODE = 2000 # Start evaluating after this many episodes
EVAL_INTERVAL = 500
NUM_EVALS = 100
LOG_INTERVAL = 100           # How often to print training progress
SAVE_INTERVAL = 1000         # How often to save the model
MODEL_SAVE_PATH = "dqn_snake_model.pth"
PARAMS_SAVE_PATH = "dqn_snake_params.bin" # Path for simplified parameter saving

DEVICE = "cuda" if torch.cuda.is_available else "cpu"
print(f"Using device: {DEVICE}")

STATE_SIZE = GRID_SIZE
ACTION_SIZE = 4