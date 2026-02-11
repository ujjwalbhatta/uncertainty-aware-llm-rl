# ULPS: Uncertainty-Aware LLM-Guided Policy Shaping (MiniGrid UnlockPickup)

This repository contains the research code and experiment artifacts used in the paper:

**"Uncertainty-Aware LLM-Guided Policy Shaping for Sparse-Reward Reinforcement Learning"**

The project implements ULPS, a framework that combines:
- A fine-tuned BERT-based LLM (action predictor)
- Monte Carlo Dropout for uncertainty estimation
- Entropy-based blending between the LLM policy and PPO policy
- Baselines including Unguided PPO, Linear Decay, Q-Learning, and DQN

## Repository Structure

- `MainCode/`  
  Main implementation of ULPS (Calibrated LLM + PPO) and experiment runners.

- `Qlearning/`  
  Q-Learning baseline implementation and related utilities.

- `DQN/`  
  DQN baseline implementation.

- `Ablation Study/`  
  Additional scripts for ablation experiments.

- `results/`  
  Pre-generated CSV logs, metrics, plots, and summaries used to create the figures/tables in the paper.

## Notes on Reproducibility

This repository is provided as a research artifact.  
The core architecture and training pipeline are included, however:

- Some results (CSV files and plots) were generated during experimentation and are included under `results/`.
- Not all result files are regenerated automatically by a single script.
- Some plotting and summarization scripts assume the same output folder structure used during the experiments.
- Several additional plots were generated during development, but only a subset was selected for the final paper for compactness.

## Running Experiments (Main ULPS)

Example scripts:
- `MainCode/run_4x4_experiment.py`
- `MainCode/run_8x8_experiment.py`

These scripts generate episode-level CSV logs and summaries.

## Output Files

The `results/` folder contains:
- episode-level logs (`episode_metrics_*.csv`)
- entropy logs (`entropy_values_*.csv`)
- calibration metrics (`calibration_metrics_*.csv`)
- final plots (`final_training_summary_*.png`)
- consolidated experiment summaries (`experiment_auc_summary.csv`)

These files were used to generate the tables and figures reported in the paper.

## Disclaimer

This codebase reflects the structure used during research and experimentation.  
Some parts may require minor path adjustments depending on your local environment.

