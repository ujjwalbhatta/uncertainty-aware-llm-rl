import os
import numpy as np
import torch
import pandas as pd
from calibrated_llm_core import config, ImprovedDQNAgent, GRPOPolicy

def setup_4x4_config():
    """Setup configuration for 4x4 environment"""
    config.TRAIN_GRID_WIDTH = 8
    config.TRAIN_GRID_HEIGHT = 8
    config.EVAL_GRID_WIDTH = 4
    config.EVAL_GRID_HEIGHT = 4
    config.MAX_STEPS = 50

    # Add parameters for improved DQN and GRPO
    config.DOUBLE_DQN = True
    config.DUELING_NETWORK = True
    config.PRIORITIZED_REPLAY = True
    config.DQN_LEARNING_RATE = 0.0005
    config.GRPO_LEARNING_RATE = 0.0003
    config.DQN_BATCH_SIZE = 64
    config.MEMORY_SIZE = 100000
    config.TARGET_UPDATE = 1000
    config.ALPHA = 0.6  # PER: prioritization exponent
    config.BETA_START = 0.4  # PER: importance sampling start value
    config.BETA_FRAMES = 100000  # PER: frames to anneal beta to 1.0
    config.EPSILON_START = 1.0
    config.EPSILON_END = 0.01
    config.EPSILON_DECAY = 500

    config.SAVE_DIR = "calibrated_llm_rl/4x4_results"
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    print("[✔] Configuration set for 4x4 environment.")


def run_dqn_grpo_improved():
    """Run Improved DQN (Double + Dueling + PER) with GRPO policy"""
    setup_4x4_config()
    
    print("🤖 Running improved DQN (Double + Dueling + PER) with GRPO policy on 4x4 grid...")
    
    # Initialize DQN agent with improvements
    dqn_agent = ImprovedDQNAgent(
        state_dim=12,  # State dimension
        action_dim=5,  # Action dimension
        lr=config.DQN_LEARNING_RATE,
        batch_size=config.DQN_BATCH_SIZE,
        memory_size=config.MEMORY_SIZE,
        gamma=config.GAMMA,
        target_update=config.TARGET_UPDATE,
        double_dqn=config.DOUBLE_DQN,
        dueling_network=config.DUELING_NETWORK,
        prioritized_replay=config.PRIORITIZED_REPLAY,
        alpha=config.ALPHA,
        beta_start=config.BETA_START,
        beta_frames=config.BETA_FRAMES,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Initialize GRPO policy
    grpo_policy = GRPOPolicy(
        dqn_agent=dqn_agent,
        lr=config.GRPO_LEARNING_RATE,
        env_width=config.EVAL_GRID_WIDTH,
        env_height=config.EVAL_GRID_HEIGHT,
        max_steps=config.MAX_STEPS,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Train the agent
    rewards, task_success = grpo_policy.train(num_episodes=10000)
    
    # Save results
    results_dir = config.SAVE_DIR
    os.makedirs(results_dir, exist_ok=True)
    rewards_path = os.path.join(results_dir, "dqn_grpo_improved_rewards.csv")
    success_path = os.path.join(results_dir, "dqn_grpo_improved_task_success.csv")
    
    rewards_df = pd.DataFrame({"episode": range(len(rewards)), "reward": rewards})
    rewards_df.to_csv(rewards_path, index=False)
    
    success_df = pd.DataFrame(task_success, columns=["key_picked", "door_opened", "goal_reached"])
    success_df.to_csv(success_path, index=False)
    
    # Calculate AUC for reward curve
    auc = np.trapz(np.array(rewards))
    print(f"DQN GRPO Improved AUC: {auc:.2f}")
    
    print(f"✅ Improved DQN+GRPO training finished. Results saved at {results_dir}")
    
    return {
        "rewards": rewards,
        "task_success": task_success,
        "auc": auc
    }

if __name__ == "__main__":
    run_dqn_grpo_improved()