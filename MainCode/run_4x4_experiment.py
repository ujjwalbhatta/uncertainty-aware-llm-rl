import os
import random
import numpy as np
import torch
from MainCode.llm_core import CalibratedRLTrainer, config

def setup_4x4_config():
    config.TRAIN_GRID_WIDTH = 8
    config.TRAIN_GRID_HEIGHT = 8
    config.EVAL_GRID_WIDTH = 4
    config.EVAL_GRID_HEIGHT = 4
    config.MAX_STEPS = 50

    config.SAVE_DIR = "calibrated_llm_rl/4x4_results"
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    print("[✔] Configuration set for 4x4 environment.")

def train_and_evaluate():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("Starting Calibrated LLM + RL on 4x4 grid...\n")
    setup_4x4_config()
    trainer = CalibratedRLTrainer()

    print("Step 1: Fine-tuning LLM on 8x8 grid...")
    accuracy = trainer.train_llm(
        env_width=config.TRAIN_GRID_WIDTH,
        env_height=config.TRAIN_GRID_HEIGHT,
        num_samples=config.TRAIN_SAMPLES
    )

    if accuracy < 0.85:
        print(f"⚠️ Accuracy too low: {accuracy:.4f}")
        return

    print("\nStep 2: Training RL on 4x4 with calibrated LLM...")
    auc = trainer.train_rl(
        env_width=config.EVAL_GRID_WIDTH,
        env_height=config.EVAL_GRID_HEIGHT
    )
    print(f"Area Under Curve (AUC): {auc:.2f}")

    print("\nStep 3: Evaluating on 4x4 grid...")
    mean_reward, task_rates = trainer.evaluate(
        env_width=config.EVAL_GRID_WIDTH,
        env_height=config.EVAL_GRID_HEIGHT,
        num_episodes=20
    )
    
    print("\nStep 4: Analyzing calibration methods...")
    results, calibration_table = trainer.evaluate_calibration_methods(
        env_width=config.EVAL_GRID_WIDTH, 
        env_height=config.EVAL_GRID_HEIGHT
    )
    
    print("\nStep 5: Visualizing incorrect guidance examples...")
    trainer.visualize_incorrect_guidance(
        env_width=config.EVAL_GRID_WIDTH,
        env_height=config.EVAL_GRID_HEIGHT
    )
    
    print("\nAll done! Results saved to:", config.SAVE_DIR)
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "mean_reward": mean_reward,
        "task_rates": task_rates,
        "calibration_results": results,
    }

def run_baseline_only():
    setup_4x4_config()
    trainer = CalibratedRLTrainer()
    
    print("Running unguided RL baseline on 4x4 grid...")
    auc = trainer.train_unguided_rl(
        env_width=config.EVAL_GRID_WIDTH,
        env_height=config.EVAL_GRID_HEIGHT
    )
    print(f"Unguided RL AUC: {auc:.2f}")
    
    return {"auc": auc}

def run_uncalibrated_only():
    setup_4x4_config()
    trainer = CalibratedRLTrainer()
    
    print("🔧 Fine-tuning LLM on 8x8 grid...")
    accuracy = trainer.train_llm(
        env_width=config.TRAIN_GRID_WIDTH,
        env_height=config.TRAIN_GRID_HEIGHT,
        num_samples=config.TRAIN_SAMPLES,
        force_finetune=False
    )
    
    print("🤖 Running uncalibrated LLM-guided RL on 4x4 grid...")
    auc = trainer.train_uncalibrated_rl(
        env_width=config.EVAL_GRID_WIDTH,
        env_height=config.EVAL_GRID_HEIGHT
    )
    print(f"Uncalibrated LLM-guided RL AUC: {auc:.2f}")
    
    return {"accuracy": accuracy, "auc": auc}

def run_linear_decay_only():
    """Run LLM-guided PPO RL with linear decay"""
    setup_4x4_config()
    trainer = CalibratedRLTrainer()
    
    print("Fine-tuning LLM on 8x8 grid...")
    accuracy = trainer.train_llm(
        env_width=config.TRAIN_GRID_WIDTH,
        env_height=config.TRAIN_GRID_HEIGHT,
        num_samples=config.TRAIN_SAMPLES,
        force_finetune=False
    )
    
    print("Running linear decay coefficient RL on 4x4 grid...")
    auc = trainer.train_linear_decay_rl(
        env_width=config.EVAL_GRID_WIDTH,
        env_height=config.EVAL_GRID_HEIGHT
    )
    print(f"Linear decay coefficient RL AUC: {auc:.2f}")
    
    return {"accuracy": accuracy, "auc": auc}

if __name__ == "__main__":
    train_and_evaluate()
