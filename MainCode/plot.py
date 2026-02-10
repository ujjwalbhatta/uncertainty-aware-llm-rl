import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# File paths
file_paths = {
    "Unguided RL 4x4": "RL_FINAL/rl_results/unguided_rl_4x4/processed_data.csv",
    "Uncalibrated RL 4x4": "RL_FINAL/rl_results/uncalibrated_rl_4x4/processed_data.csv",
    "Linear RL 4x4": "RL_FINAL/rl_results/linear_rl_4x4/processed_data.csv",  
    "DQN": "RL_FINAL/rl_results/dqn/processed_data.csv",
    "Q-Learning": "RL_FINAL/rl_results/qlearning/processed_episodes.csv",
    "Calibrated LLM RL 4x4": "RL_FINAL/calibrated_llm_rl/4x4_results/episode_metrics_4x4.csv",
    "Calibrated LLM RL 8x8": "RL_FINAL/calibrated_llm_rl/8x8_results/episode_metrics_8x8.csv"
}
if not os.path.exists(file_paths["Linear RL 4x4"]):
    file_paths["Linear RL 4x4"] += "/processed_data.csv"

# Function to read and preprocess each dataset
def load_experiment_data(file_path, experiment_name):
    try:
        df = pd.read_csv(file_path)
        
        # Add experiment name column
        df['experiment'] = experiment_name
        
        # Handle different formats with boolean vs 0/1 for goal/key/door
        if 'goal_reached' in df.columns:  # Convert boolean to numeric if needed
            df['goal'] = df['goal_reached'].map({True: 1, False: 0}) if df['goal_reached'].dtype == bool else df['goal_reached']
        
        if 'key_picked' in df.columns:
            df['key'] = df['key_picked'].map({True: 1, False: 0}) if df['key_picked'].dtype == bool else df['key_picked']
            
        if 'door_opened' in df.columns:
            df['door'] = df['door_opened'].map({True: 1, False: 0}) if df['door_opened'].dtype == bool else df['door_opened']
            
        # For cumulative performance analysis
        df = df.sort_values('episode')
        
        return df
    except Exception as e:
        print(f"Error loading {experiment_name}: {e}")
        return None

# Load all datasets
datasets = {}
for name, path in file_paths.items():
    datasets[name] = load_experiment_data(path, name)

# Function to calculate cumulative success rate (for AUC-like metric)
def calculate_performance_metrics(datasets):
    results = {}
    
    for name, df in datasets.items():
        if df is not None:
            # Calculate cumulative success rate (goal reached)
            # This will serve as our "AUC" - higher area = better performance
            df['cumulative_success'] = df['goal'].cumsum() / (df.index + 1)
            
            # Calculate average reward
            df['avg_reward_so_far'] = df['reward'].cumsum() / (df.index + 1)
            
            results[name] = df
    
    return results

# Calculate metrics
performance_results = calculate_performance_metrics(datasets)

# Function to plot performance curves
def plot_performance_curves(results, metric='avg_reward_so_far', title='Learning Performance'):
    plt.figure(figsize=(14, 10))
    
    for name, df in results.items():
        if df is not None and metric in df.columns:
            plt.plot(df['episode'], df[metric], label=name, linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Calculate AUC for each experiment using trapezoidal rule
    aucs = {}
    for name, df in results.items():
        if df is not None and metric in df.columns:
            # Calculate AUC using numpy's trapz
            auc = np.trapz(df[metric], df['episode'])
            aucs[name] = auc
    
    # Add AUC values to the plot
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = [f"{label} (AUC: {aucs[label]:.2f})" for label in labels]
    plt.legend(handles, labels, fontsize=10)
    
    plt.tight_layout()
    return plt, aucs

# Plot reward curves - primary focus now
reward_plot, reward_aucs = plot_performance_curves(performance_results, 
                                                 metric='avg_reward_so_far', 
                                                 title='Learning Performance: Average Reward')
reward_plot.savefig('reward_auc.png')

# Plot cumulative success rate as secondary
success_plot, success_aucs = plot_performance_curves(performance_results, 
                                                    metric='cumulative_success', 
                                                    title='Learning Performance: Goal Success Rate')
success_plot.savefig('success_rate_auc.png')

# Bar plot for Reward AUC comparison
plt.figure(figsize=(14, 10))

# Sort experiments by Reward AUC value (changed from success_aucs to reward_aucs)
sorted_aucs = dict(sorted(reward_aucs.items(), key=lambda item: item[1], reverse=True))
experiments = list(sorted_aucs.keys())
auc_values = list(sorted_aucs.values())

# Create bar plot
bars = plt.bar(experiments, auc_values, color=sns.color_palette('viridis', len(experiments)))
plt.xticks(rotation=45, ha='right')
plt.title('Reward AUC Comparison Across Experiments', fontsize=16)
plt.ylabel('Area Under Curve (Average Reward)', fontsize=14)
plt.grid(axis='y', alpha=0.3)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.2f}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('reward_auc_comparison.png')

# Create a summary DataFrame
summary_df = pd.DataFrame({
    'Experiment': experiments,
    'Reward AUC': [reward_aucs[exp] for exp in experiments],
    'Success Rate AUC': [success_aucs.get(exp, np.nan) for exp in experiments]
}).sort_values('Reward AUC', ascending=False)

print("AUC Summary:")
print(summary_df.to_string(index=False))

# Save summary to CSV
summary_df.to_csv('experiment_auc_summary.csv', index=False)

# 1. Success rate (total wins / number of episodes)
def plot_success_rate(results):
    plt.figure(figsize=(14, 10))
    
    # Prepare data for proper bar positioning
    success_rates = []
    experiment_names = []
    
    for name, df in results.items():
        if df is not None and 'goal' in df.columns:
            # Calculate success rate per episode
            success_rate = df['goal'].mean() * 100
            success_rates.append(success_rate)
            experiment_names.append(name)
    
    # Sort by success rate (descending)
    sorted_indices = np.argsort(success_rates)[::-1]
    sorted_names = [experiment_names[i] for i in sorted_indices]
    sorted_rates = [success_rates[i] for i in sorted_indices]
    
    # Plot bars
    bars = plt.bar(sorted_names, sorted_rates, color=sns.color_palette('viridis', len(sorted_names)))
    
    plt.title('Success Rate by Experiment', fontsize=16)
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2, 
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('success_rate_by_experiment.png')
    return plt

# 2. Average steps to goal (only for successful episodes)
def plot_avg_steps_to_goal(results):
    plt.figure(figsize=(14, 10))
    avg_steps = []
    experiment_names = []
    
    for name, df in results.items():
        if df is not None:
            steps_col = 'length' if 'length' in df.columns else 'steps'
            
            if steps_col in df.columns and 'goal' in df.columns:
                # Filter only successful episodes
                successful_df = df[df['goal'] == 1]
                if len(successful_df) > 0:
                    avg_step = successful_df[steps_col].mean()
                    avg_steps.append(avg_step)
                    experiment_names.append(name)
    
    # Sort by average steps (ascending - lower is better)
    sorted_indices = np.argsort(avg_steps)
    sorted_names = [experiment_names[i] for i in sorted_indices]
    sorted_steps = [avg_steps[i] for i in sorted_indices]
    
    # Plot
    bars = plt.bar(sorted_names, sorted_steps, color=sns.color_palette('viridis', len(sorted_names)))
    plt.title('Average Steps to Goal (Successful Episodes Only)', fontsize=16)
    plt.ylabel('Number of Steps', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('avg_steps_to_goal.png')
    return plt

# 3. Average reward per episode
def plot_avg_reward(results):
    plt.figure(figsize=(14, 10))
    avg_rewards = []
    experiment_names = []
    
    for name, df in results.items():
        if df is not None and 'reward' in df.columns:
            avg_reward = df['reward'].mean()
            avg_rewards.append(avg_reward)
            experiment_names.append(name)
    
    # Sort by average reward (descending - higher is better)
    sorted_indices = np.argsort(avg_rewards)[::-1]
    sorted_names = [experiment_names[i] for i in sorted_indices]
    sorted_rewards = [avg_rewards[i] for i in sorted_indices]
    
    # Plot
    bars = plt.bar(sorted_names, sorted_rewards, color=sns.color_palette('viridis', len(sorted_names)))
    plt.title('Average Reward per Episode', fontsize=16)
    plt.ylabel('Average Reward', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('avg_reward_per_episode.png')
    return plt

# 4. Total wins and total steps - Fixed version
def plot_total_wins_steps(results):
    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(14, 10))
    
    # Set up data
    total_wins = []
    total_steps = []
    experiment_names = []
    
    for name, df in results.items():
        if df is not None and 'goal' in df.columns:
            steps_col = 'length' if 'length' in df.columns else 'steps'
            
            if steps_col in df.columns:
                wins = df['goal'].sum()
                steps = df[steps_col].sum()
                
                total_wins.append(wins)
                total_steps.append(steps)
                experiment_names.append(name)
    
    # Sort by total wins (descending)
    sorted_indices = np.argsort(total_wins)[::-1]
    sorted_names = [experiment_names[i] for i in sorted_indices]
    sorted_wins = [total_wins[i] for i in sorted_indices]
    sorted_steps = [total_steps[i] for i in sorted_indices]
    
    # Set up x positions
    x = np.arange(len(sorted_names))
    width = 0.35
    
    # Create second y-axis
    ax2 = ax1.twinx()
    
    # Plot wins on first axis
    bars1 = ax1.bar(x - width/2, sorted_wins, width, color='blue', alpha=0.7, label='Total Wins')
    ax1.set_ylabel('Total Wins', color='blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot steps on second axis
    bars2 = ax2.bar(x + width/2, sorted_steps, width, color='red', alpha=0.7, label='Total Steps')
    ax2.set_ylabel('Total Steps', color='red', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # X-axis setup
    ax1.set_title('Total Wins vs Total Steps by Experiment', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_names, rotation=45, ha='right')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Add value labels
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='blue', fontsize=12)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='red', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('total_wins_steps.png')
    return plt

# 5. Success rate over training (moving average) - use maximum possible window
def plot_success_rate_over_training(results, window=250):  # Set to 250 for maximum smoothing
    plt.figure(figsize=(16, 12))  # Larger figure size
    
    for name, df in results.items():
        if df is not None and 'goal' in df.columns and 'episode' in df.columns:
            # Calculate rolling success rate
            df = df.sort_values('episode')
            # Use min(len(df), window) to ensure window isn't larger than dataset
            actual_window = min(len(df), window)
            df['rolling_success'] = df['goal'].rolling(window=actual_window, min_periods=1).mean() * 100
            
            plt.plot(df['episode'], df['rolling_success'], label=f"{name} (window={actual_window})", linewidth=2)
    
    plt.title(f'Success Rate Over Training (Maximum Window Size)', fontsize=18)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Success Rate (%)', fontsize=16)
    plt.legend(fontsize=14, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)  # 0-100% with a little margin
    
    plt.tight_layout()
    plt.savefig('success_rate_over_training.png')
    return plt

# 6. Average steps to goal over training (for all episodes) - use maximum possible window
def plot_avg_steps_over_training(results, window=250):  # Set to 250 for maximum smoothing
    plt.figure(figsize=(16, 12))  # Larger figure size
    
    for name, df in results.items():
        if df is not None and 'episode' in df.columns:
            steps_col = 'length' if 'length' in df.columns else 'steps'
            
            if steps_col in df.columns:
                # Calculate rolling average steps
                df = df.sort_values('episode')
                # Use min(len(df), window) to ensure window isn't larger than dataset
                actual_window = min(len(df), window)
                df['rolling_steps'] = df[steps_col].rolling(window=actual_window, min_periods=1).mean()
                
                plt.plot(df['episode'], df['rolling_steps'], label=f"{name} (window={actual_window})", linewidth=2)
    
    plt.title(f'Average Steps Over Training (Maximum Window Size)', fontsize=18)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Number of Steps', fontsize=16)
    plt.legend(fontsize=14, loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('avg_steps_over_training.png')
    return plt

# 7. Average reward over training (moving average) - use maximum possible window
def plot_avg_reward_over_training(results, window=250):  # Set to 250 for maximum smoothing
    plt.figure(figsize=(16, 12))  # Larger figure size
    
    for name, df in results.items():
        if df is not None and 'reward' in df.columns and 'episode' in df.columns:
            # Calculate rolling average reward
            df = df.sort_values('episode')
            # Use min(len(df), window) to ensure window isn't larger than dataset
            actual_window = min(len(df), window)
            df['rolling_reward'] = df['reward'].rolling(window=actual_window, min_periods=1).mean()
            
            plt.plot(df['episode'], df['rolling_reward'], label=f"{name} (window={actual_window})", linewidth=2)
    
    plt.title(f'Average Reward Over Training (Maximum Window Size)', fontsize=18)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Average Reward', fontsize=16)
    plt.legend(fontsize=14, loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('avg_reward_over_training.png')
    return plt

# Create all the plots
success_rate_plot = plot_success_rate(performance_results)
avg_steps_plot = plot_avg_steps_to_goal(performance_results)
avg_reward_plot = plot_avg_reward(performance_results)
total_stats_plot = plot_total_wins_steps(performance_results)  # Fixed plot
success_rate_training_plot = plot_success_rate_over_training(performance_results, window=250)  # Maximum window
steps_training_plot = plot_avg_steps_over_training(performance_results, window=250)  # Maximum window
reward_training_plot = plot_avg_reward_over_training(performance_results, window=250)  # Maximum window

# Create a comprehensive summary DataFrame with all metrics
summary_metrics = []

for name, df in performance_results.items():
    if df is not None and 'goal' in df.columns:
        steps_col = 'length' if 'length' in df.columns else 'steps'
        
        if steps_col in df.columns:
            # Calculate all metrics
            success_rate = df['goal'].mean() * 100
            
            # Average steps to goal (only successful episodes)
            successful_df = df[df['goal'] == 1]
            avg_steps_to_goal = successful_df[steps_col].mean() if len(successful_df) > 0 else np.nan
            
            # Other metrics
            avg_reward = df['reward'].mean()
            total_wins = df['goal'].sum()
            total_steps = df[steps_col].sum()
            
            # AUC metrics (already calculated earlier)
            reward_auc = reward_aucs.get(name, np.nan)
            success_auc = success_aucs.get(name, np.nan)
            
            # Store all metrics
            summary_metrics.append({
                'Experiment': name,
                'Success Rate (%)': success_rate,
                'Average Steps to Goal': avg_steps_to_goal,
                'Average Reward': avg_reward,
                'Total Wins': total_wins,
                'Total Steps': total_steps,
                'Reward AUC': reward_auc,
                'Success Rate AUC': success_auc
            })

# Create comprehensive summary DataFrame - sort by Reward AUC (changed from Success Rate)
comprehensive_summary = pd.DataFrame(summary_metrics).sort_values('Reward AUC', ascending=False)

print("\nComprehensive Summary:")
print(comprehensive_summary.to_string(index=False))

# Save comprehensive summary to CSV
comprehensive_summary.to_csv('comprehensive_experiment_summary.csv', index=False)

# Create a radar chart to compare experiments across multiple metrics - Updated to use Reward AUC
def create_radar_chart(summary_df):
    # Select metrics for radar chart
    metrics = ['Success Rate (%)', 'Average Steps to Goal', 'Average Reward', 'Reward AUC']  # Changed to Reward AUC
    
    # Normalize values for radar chart (0-1 scale)
    radar_data = summary_df.copy()
    
    # Invert 'Average Steps to Goal' since lower is better
    if 'Average Steps to Goal' in radar_data.columns:
        max_steps = radar_data['Average Steps to Goal'].max()
        radar_data['Average Steps to Goal'] = 1 - (radar_data['Average Steps to Goal'] / max_steps)
    
    # Normalize other metrics
    for metric in metrics:
        if metric != 'Average Steps to Goal' and metric in radar_data.columns:
            min_val = radar_data[metric].min()
            max_val = radar_data[metric].max()
            if max_val > min_val:
                radar_data[metric] = (radar_data[metric] - min_val) / (max_val - min_val)
    
    # Number of variables
    N = len(metrics)
    
    # Create angle for each variable
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 14))  # Larger figure size
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], metrics, size=14)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=12)
    plt.ylim(0, 1)
    
    # Plot each experiment
    for i, exp in enumerate(radar_data['Experiment']):
        values = radar_data.loc[radar_data['Experiment'] == exp, metrics].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=exp)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=14)
    plt.title('Experiment Performance Comparison', size=18)
    
    plt.tight_layout()
    plt.savefig('radar_chart_comparison.png')
    return plt

# Create radar chart
radar_plot = create_radar_chart(comprehensive_summary)

plt.show()