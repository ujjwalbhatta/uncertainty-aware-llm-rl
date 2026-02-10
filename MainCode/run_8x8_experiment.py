import os
import random
import numpy as np
import torch
from MainCode.llm_core import CalibratedRLTrainer, config

def setup_8x8_config():
    """Setup configuration for 8x8 environment"""
    # Environment settings
    config.TRAIN_GRID_WIDTH = 8
    config.TRAIN_GRID_HEIGHT = 8
    config.EVAL_GRID_WIDTH = 8
    config.EVAL_GRID_HEIGHT = 8
    config.MAX_STEPS = 100  # Increased for larger environment
    
    # Dataset settings
    config.TRAIN_SAMPLES = 25000  # Increase samples for larger environment

    # Save directory
    config.SAVE_DIR = "calibrated_llm_rl/8x8_results"
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    print("[✔] Configuration set for 8x8 environment.")

def train_and_evaluate():
    """Train and evaluate model on 8x8 environment"""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("🧠 Starting Calibrated LLM + RL on 8x8 grid...\n")
    setup_8x8_config()
    trainer = CalibratedRLTrainer()

    # Step 1: Fine-tune LLM on 8x8 grid
    print("🔧 Step 1: Fine-tuning LLM on 8x8 grid...")
    accuracy = trainer.train_llm(
        env_width=config.TRAIN_GRID_WIDTH,
        env_height=config.TRAIN_GRID_HEIGHT,
        num_samples=config.TRAIN_SAMPLES,
        force_finetune=False
    )

    if accuracy < 0.85:
        print(f"⚠️ Accuracy too low: {accuracy:.4f}. Consider increasing samples or tweaking prompt.")
        return

    # Step 2: Training RL with calibrated LLM
    print("\n🤖 Step 2: Training RL on 8x8 with calibrated LLM...")
    auc = trainer.train_rl(
        env_width=config.EVAL_GRID_WIDTH,
        env_height=config.EVAL_GRID_HEIGHT
    )
    print(f"Area Under Curve (AUC): {auc:.2f}")

    # Step 3: Evaluation
    print("\n📊 Step 3: Evaluating on 8x8 grid...")
    mean_reward, task_rates = trainer.evaluate(
        env_width=config.EVAL_GRID_WIDTH,
        env_height=config.EVAL_GRID_HEIGHT,
        num_episodes=20
    )
    
    # Step 4: Calibration Method Analysis
    print("\n🔍 Step 4: Analyzing calibration methods...")
    results, calibration_table = trainer.evaluate_calibration_methods(
        env_width=config.EVAL_GRID_WIDTH, 
        env_height=config.EVAL_GRID_HEIGHT
    )
    
    # Step 5: Visualize incorrect guidance examples
    print("\n🖼️ Step 5: Visualizing incorrect guidance examples...")
    trainer.visualize_incorrect_guidance(
        env_width=config.EVAL_GRID_WIDTH,
        env_height=config.EVAL_GRID_HEIGHT
    )
    
    print("\n✅ All done! Results saved to:", config.SAVE_DIR)
    
    # Return summary of results
    return {
        "accuracy": accuracy,
        "auc": auc,
        "mean_reward": mean_reward,
        "task_rates": task_rates,
        "calibration_results": results
    }

if __name__ == "__main__":
    train_and_evaluate()