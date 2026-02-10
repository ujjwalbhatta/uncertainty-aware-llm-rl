import os
from run_4x4_experiment_improved import run_dqn_grpo_improved

def main():
    os.makedirs("calibrated_llm_rl", exist_ok=True)
    print("=" * 80)
    print("Double DQN with Dueling Architecture and PER for Reinforcement Learning")
    print("=" * 80)
    
    print("\nRunning Double DQN + Dueling + PER with GRPO policy...")
    results = run_dqn_grpo_improved()
    
    print("\n" + "=" * 80)
    print(f"AUC: {results['auc']:.2f}")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    main()