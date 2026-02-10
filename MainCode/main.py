import os
import argparse
import time
from run_4x4_experiment import train_and_evaluate as train_4x4
from run_4x4_experiment import run_baseline_only as baseline_4x4
from run_4x4_experiment import run_uncalibrated_only as uncalibrated_4x4
from run_4x4_experiment import run_linear_decay_only as linear_decay_4x4
from run_8x8_experiment import train_and_evaluate as train_8x8

def parse_args():
    parser = argparse.ArgumentParser(description="Run Calibrated LLM RL Experiments")
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', '4x4', '8x8', 'analysis', 'baseline4x4', 
                                 'uncalibrated4x4', 'linear4x4'],  
                        help='Which experiment to run')
    parser.add_argument('--no-analysis', action='store_true',
                        help='Skip running the analysis after experiments')
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    
    print("=" * 80)
    print("🧠 Calibrated LLM for Reinforcement Learning")
    print("=" * 80)
    
    os.makedirs("calibrated_llm_rl", exist_ok=True)
    
    results = {}
    
    if args.mode in ['all', '4x4']:
        print("\nRunning 4x4 experiment...")
        results['4x4'] = train_4x4()
        
    if args.mode in ['all', '8x8']:
        print("\nRunning 8x8 experiment...")
        results['8x8'] = train_8x8()
        
    if args.mode == 'baseline4x4':
        print("\nRunning 4x4 baseline only...")
        results['baseline'] = baseline_4x4()
        
    if args.mode == 'uncalibrated4x4':
        print("\nRunning 4x4 uncalibrated only...")
        results['uncalibrated'] = uncalibrated_4x4()
        
    if args.mode == 'linear4x4':
        print("\nRunning 4x4 linear decay only...")
        results['linear'] = linear_decay_4x4()
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 80)
    print(f"All done! Total runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    main()
