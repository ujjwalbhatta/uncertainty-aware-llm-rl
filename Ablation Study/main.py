import os
import argparse
import time
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Run MC Dropout Ablation Study")
    parser.add_argument('--dropout-rates', nargs='+', type=float, 
                       default=[0.05, 0.1, 0.2],
                       help='Dropout rates to test')
    parser.add_argument('--forward-passes', nargs='+', type=int, 
                       default=[4, 8, 12],
                       help='Number of forward passes to test')
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    
    print("=" * 80)
    print("MC Dropout Ablation Study for Calibrated LLM RL")
    print("=" * 80)
    
    os.makedirs("calibrated_llm_rl", exist_ok=True)
    
    # Run ablation study
    results_df = run_ablation_study(args.dropout_rates, args.forward_passes)
    
    # Save results
    results_df.to_csv("calibrated_llm_rl/ablation_results.csv", index=False)
    
    # Display results table
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    
    return results_df

if __name__ == "__main__":
    main()