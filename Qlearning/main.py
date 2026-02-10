# main.py

import os
import random
import numpy as np
import torch
from config import Config
from envs import UnlockPickupEnv
from oracle import QlearningOracle
from trainer import CalibratedRLTrainer
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed()
    config = Config()
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    print("=" * 80)
    print(f"🎯 Starting Calibrated LLM + Q-learning Oracle + GRPO Training")
    print("=" * 80)

    # Step 1: Train or Load Oracle
    print("\n0️⃣ Pre-training Q-learning Oracle...")
    env = UnlockPickupEnv(width=config.TRAIN_GRID_WIDTH, height=config.TRAIN_GRID_HEIGHT, max_steps=config.MAX_STEPS)
    oracle = QlearningOracle(env_width=config.TRAIN_GRID_WIDTH, env_height=config.TRAIN_GRID_HEIGHT)

    oracle_path = os.path.join(config.SAVE_DIR, f"oracle_{config.TRAIN_GRID_WIDTH}x{config.TRAIN_GRID_HEIGHT}.pkl")
    if os.path.exists(oracle_path):
        print(f"📦 Loading existing Oracle from {oracle_path}")
        oracle.load(oracle_path)
    else:
        print(f"🧠 Training Oracle from scratch for {config.QLEARNING_EPISODES} episodes...")
        oracle.train(env, num_episodes=config.QLEARNING_EPISODES)
        oracle.save(oracle_path)

    # Step 2: Fine-tune or Load LLM
    print("\n1️⃣ Fine-tuning or Loading Calibrated LLM...")
    trainer = CalibratedRLTrainer()

    llm_path = os.path.join(config.SAVE_DIR, "best_llm.pt")
    if os.path.exists(llm_path):
        print(f"📦 Loading existing fine-tuned LLM from {llm_path}")
        trainer.llm.load_state_dict(torch.load(llm_path, map_location=trainer.device))
    else:
        print(f"🧠 Fine-tuning LLM...")
        trainer.train_llm(
            oracle,
            width=config.TRAIN_GRID_WIDTH,
            height=config.TRAIN_GRID_HEIGHT,
            force_finetune=True
        )

    # Step 3: Train RL Policy
    print("\n2️⃣ Training RL Policy with GRPO + Curriculum...")
    trainer.train_rl()

    # Step 4: Evaluate Policy
    print("\n3️⃣ Evaluating final trained agent...")
    evaluate_policy(trainer, config)

def evaluate_policy(trainer, config):
    """
    Evaluate the trained policy.
    """
    env = UnlockPickupEnv(width=config.EVAL_GRID_WIDTH, height=config.EVAL_GRID_HEIGHT, max_steps=config.MAX_STEPS)

    policy_path = os.path.join(config.SAVE_DIR, "final_policy.pt")
    if os.path.exists(policy_path):
        trainer.policy.load_state_dict(torch.load(policy_path, map_location=trainer.device))

    num_eval_episodes = 100
    rewards = []
    steps = []
    task_success = {"key": 0, "door": 0, "goal": 0}

    print(f"🏁 Running {num_eval_episodes} evaluation episodes...")
    for episode in range(num_eval_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            prompt = trainer.prompt_gen.generate(env)
            state_emb = trainer.llm.get_embedding(prompt).unsqueeze(0).to(trainer.device)

            llm_probs, uncertainty = trainer.llm.mc_predict(prompt)
            llm_probs = llm_probs.to(trainer.device)

            action, _, _, _, _ = trainer.policy.get_action(
                state_emb,
                llm_probs=llm_probs,
                uncertainty=uncertainty
            )

            next_state, reward, done, truncated, _ = env.step(action.item())
            done = done or truncated or (step_count >= config.MAX_STEPS - 1)

            total_reward += reward
            step_count += 1
            state = next_state

        rewards.append(total_reward)
        steps.append(step_count)

        if env.key_picked:
            task_success["key"] += 1
        if env.door_opened:
            task_success["door"] += 1
        if env.agent_pos == env.goal_pos and env.door_opened:
            task_success["goal"] += 1

        if (episode + 1) % 10 == 0:
            print(f"✅ Completed {episode+1}/{num_eval_episodes} episodes...")

    # Final results
    print("\n📊 Evaluation Summary:")
    print(f"  Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Avg Steps: {np.mean(steps):.2f} ± {np.std(steps):.2f}")
    print(f"  Key Pickup Success Rate: {task_success['key'] / num_eval_episodes:.2%}")
    print(f"  Door Open Success Rate: {task_success['door'] / num_eval_episodes:.2%}")
    print(f"  Goal Reached Success Rate: {task_success['goal'] / num_eval_episodes:.2%}")

    eval_df = pd.DataFrame({
        "episode": range(num_eval_episodes),
        "reward": rewards,
        "steps": steps,
        "key_picked": [1 if i < task_success["key"] else 0 for i in range(num_eval_episodes)],
        "door_opened": [1 if i < task_success["door"] else 0 for i in range(num_eval_episodes)],
        "goal_reached": [1 if i < task_success["goal"] else 0 for i in range(num_eval_episodes)],
    })
    eval_df.to_csv(os.path.join(config.SAVE_DIR, "evaluation_results.csv"), index=False)
    print(f"✅ Saved evaluation results to {config.SAVE_DIR}/evaluation_results.csv")

if __name__ == "__main__":
    main()
