<<<<<<< HEAD
import os
=======
>>>>>>> 223ec4294ea6fc8946955f814b2169544d89bf6c
import torch
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
<<<<<<< HEAD
import random as rd
=======
>>>>>>> 223ec4294ea6fc8946955f814b2169544d89bf6c
import time

import config
from game import SnakeGame
from agent import DQNAgent
<<<<<<< HEAD

def eval(env: SnakeGame, agent: DQNAgent, episode: int, interact: bool):
    print("Starting evaluation...")
    
    score = 0
    kills = 0
    death = 0
    rounds = 0
    steps = 0
    
    for i in range(config.NUM_EVALS):
        rounds += 1
        state = env.genBoard() # Initial state tensor (H, W, C)
        
        step = 0
        action = agent.select_action(state, env, True)
        enemy_actions = [agent.select_dumb_action(frame.state, env) for frame in env.enemies]
        next_state, reward, done = env.step(action, enemy_actions)
        state = next_state.clone()
        step += 1
        if i + 1 == config.NUM_EVALS and interact:
            env.print(action)
            input()
        
        while not done and step < config.MAX_STEPS_PER_EPISODE:
            action = agent.select_action(next_state, env, True)
            enemy_actions = [agent.select_dumb_action(frame.state, env) for frame in env.enemies]
            next_state, reward, done = env.step(action, enemy_actions)
            state = next_state.clone()
            step += 1
            if i + 1 == config.NUM_EVALS and interact:
                env.print(action)
                input()
                
        score += env.score
        kills += env.kill
        death += env.dead
            
        steps += step
        
    print(f"Evaluation finished. Episode {episode} Score: {score / rounds:.2f} Kills: {kills / rounds:.2f} Deaths: {death / rounds:.2f} Steps: {steps / rounds:.2f}")


def train():
    print("Configuration settings:")
    for key, value in vars(config).items():
        if not key.startswith("__") and not callable(value):
            print(f"{key}: {value}")
    seed = config.SEED
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = SnakeGame(grid_size=config.GRID_SIZE)
    agent = DQNAgent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE)
    
    if config.MODEL_PATH != None:
        agent.policy_net.load_state_dict(torch.load(config.MODEL_PATH))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

    episode_lengths = []
    episode_lengths_window = deque(maxlen=100)
    epsilons = []
    losses = []                     # List to store losses for each episode
    losses_window = deque(maxlen=100) # Last 100 losses for moving average
=======
from utils import save_simple_params

def train():
    print("Starting training...")
    print(f"Configuration: {config.DEVICE}, LR={config.LR}, BATCH={config.BATCH_SIZE}, GAMMA={config.GAMMA}")

    env = SnakeGame(grid_size=config.GRID_SIZE)
    agent = DQNAgent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, seed=0)

    scores = []                     # List containing scores from each episode
    scores_window = deque(maxlen=100) # Last 100 scores for moving average
    episode_lengths = []
    episode_lengths_window = deque(maxlen=100)
    epsilons = []
>>>>>>> 223ec4294ea6fc8946955f814b2169544d89bf6c

    start_time = time.time()

    for i_episode in range(1, config.NUM_EPISODES + 1):
<<<<<<< HEAD
        state = env.genBoard() # Initial state tensor (H, W, C)
        score = 0
        ep_len = 0
        episode_loss = 0.0

        for t in range(config.MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state, env)
            enemy_actions = [agent.select_action(frame.state, env, True) for frame in env.enemies]
            next_state, reward, done = env.step(action, enemy_actions)

            # Agent learns from the experience and returns loss
            loss = agent.step(state, action, reward, next_state, done)
            if loss is not None:
                episode_loss += loss

            state = next_state.clone()
            if reward == 10:
                score += 1 # Using raw reward here, could also use game.score
=======
        state = env.reset() # Initial state tensor (H, W, C)
        score = 0
        ep_len = 0

        for t in range(config.MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            # Agent learns from the experience
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward # Using raw reward here, could also use game.score
>>>>>>> 223ec4294ea6fc8946955f814b2169544d89bf6c
            ep_len += 1

            if done:
                break

<<<<<<< HEAD
        episode_lengths_window.append(ep_len)
        episode_lengths.append(ep_len)
        epsilons.append(agent.epsilon)    # Record epsilon evolution
        losses_window.append(episode_loss)
        losses.append(episode_loss)
=======
        scores_window.append(score)       # Save most recent score
        scores.append(score)              # Save most recent score
        episode_lengths_window.append(ep_len)
        episode_lengths.append(ep_len)
        epsilons.append(agent.epsilon)    # Record epsilon evolution
>>>>>>> 223ec4294ea6fc8946955f814b2169544d89bf6c

        # Print progress
        if i_episode % config.LOG_INTERVAL == 0:
            elapsed_time = time.time() - start_time
<<<<<<< HEAD
            avg_len = np.mean(episode_lengths_window)
            avg_loss = np.mean(losses_window)
            print(f'Episode {i_episode}\tAvg Len: {avg_len:.1f}\tAvg Loss: {avg_loss:.4f}\tEpsilon: {agent.epsilon:.4f}\tTime: {elapsed_time:.1f}s')
            start_time = time.time() # Reset timer for next interval
            
            if i_episode >= config.EVAL_START_EPISODE and i_episode % config.EVAL_INTERVAL == 0:
                eval(env, agent, i_episode, config.INTERACT) # Evaluate the agent every LOG_INTERVAL episodes

        if i_episode % config.SAVE_INTERVAL == 0:
            agent.save_model(os.path.join("models", os.path.join(config.GAME_MODE, (config.MODEL_SAVE_PATH + f"_{i_episode}.pth"))))

    print("Training finished.")

def convert_model_to_onnx():
    model = torch.load(config.MODEL_PATH)
    agent = DQNAgent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE)
    agent.policy_net.load_state_dict(model)
    dummy_input = torch.randn(1, config.GRID_SIZE, config.GRID_SIZE, config.STATE_SIZE)  # Adjust the input size as needed
    torch.onnx.export(model, dummy_input, (config.ONNX_EXPORT_PATH), export_params=True)

    print(f"Model converted to ONNX format and saved at {config.ONNX_EXPORT_PATH}")

if __name__ == "__main__":
    # train()
    convert_model_to_onnx()
=======
            avg_score = np.mean(scores_window)
            avg_len = np.mean(episode_lengths_window)
            print(f'Episode {i_episode}\tAvg Score: {avg_score:.2f}\tAvg Len: {avg_len:.1f}\tEpsilon: {agent.epsilon:.4f}\tTime: {elapsed_time:.1f}s')
            start_time = time.time() # Reset timer for next interval

        # Save model periodically
        if i_episode % config.SAVE_INTERVAL == 0:
            agent.save_model(config.MODEL_SAVE_PATH)
            # Save simplified parameters for non-Python use
            save_simple_params(agent.policy_net, config.PARAMS_SAVE_PATH)

    print("Training finished.")
    # Save final model
    agent.save_model(config.MODEL_SAVE_PATH)
    save_simple_params(agent.policy_net, config.PARAMS_SAVE_PATH)

    # Plotting (optional)
    plot_performance(scores, episode_lengths, epsilons)


def plot_performance(scores, lengths, epsilons):
    """Plots training scores, episode lengths, and epsilon decay."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot scores
    axs[0].plot(np.arange(len(scores)), scores, label='Episode Score')
    moving_avg_scores = np.convolve(scores, np.ones(100)/100, mode='valid')
    axs[0].plot(np.arange(len(moving_avg_scores)) + 99, moving_avg_scores, label='Moving Avg (100 episodes)', color='red')
    axs[0].set_ylabel('Score')
    axs[0].set_title('Episode Scores over Time')
    axs[0].legend()
    axs[0].grid(True)

    # Plot episode lengths
    axs[1].plot(np.arange(len(lengths)), lengths, label='Episode Length')
    moving_avg_lengths = np.convolve(lengths, np.ones(100)/100, mode='valid')
    axs[1].plot(np.arange(len(moving_avg_lengths)) + 99, moving_avg_lengths, label='Moving Avg (100 episodes)', color='red')
    axs[1].set_ylabel('Steps')
    axs[1].set_title('Episode Lengths over Time')
    axs[1].legend()
    axs[1].grid(True)

     # Plot epsilon decay
    axs[2].plot(np.arange(len(epsilons)), epsilons, label='Epsilon Value', color='green')
    axs[2].set_xlabel('Episode #')
    axs[2].set_ylabel('Epsilon')
    axs[2].set_title('Epsilon Decay')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig("training_performance.png")
    print("Performance plot saved to training_performance.png")
    # plt.show()

if __name__ == "__main__":
    train()
>>>>>>> 223ec4294ea6fc8946955f814b2169544d89bf6c
