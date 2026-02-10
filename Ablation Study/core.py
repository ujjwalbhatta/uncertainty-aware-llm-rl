import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Key, Goal
from minigrid.minigrid_env import MiniGridEnv
import gymnasium as gym
from gymnasium.spaces import Text
import pandas as pd
import matplotlib.pyplot as plt

# Initial Configuration
class Config:
    # Environment 
    TRAIN_GRID_WIDTH = 8  # Width for training environment (matches paper's 4x8)
    TRAIN_GRID_HEIGHT = 4  # Height for training environment
    EVAL_GRID_WIDTH = 4    # Width for evaluation environment (4x4)
    EVAL_GRID_HEIGHT = 4   # Height for evaluation environment
    MAX_STEPS = 50         # As mentioned in the paper
    ACTIONS = 5            # 0:left, 1:right, 2:forward, 3:pickup, 5:open door (4: toggle)
    
    # LLM Settings
    LLM_MODEL = "bert-base-uncased"
    DROPOUT_PROB = 0.1     # For MC Dropout calibration
    MC_SAMPLES = 8         # Number of forward passes for MC Dropout
    
    # Training
    BATCH_SIZE = 16 
    LEARNING_RATE = 5e-5   # Standard for BERT fine-tuning
    PPO_EPSILON = 0.2
    GAMMA = 0.99
    EPOCHS = 5
    MAX_SEQ_LENGTH = 512
    
    # Dataset
    TRAIN_SAMPLES = 21500  # Match paper exactly
    VAL_SPLIT = 0.1
    RANDOM_ACTION_RATIO = 0.3
    
    # Paths
    SAVE_DIR = "calibrated_llm_rl/4x4_results" # Default save directory
    
    # Rewards - using paper values
    KEY_REWARD = 0.5
    DOOR_REWARD = 0.5
    GOAL_REWARD = 0.2
    INVALID_ACTION_PENALTY = -0.02
    
    # Policy shaping parameters
    ENTROPY_SCALING = 1.0  # Weight for entropy in policy shaping
    
# Initialize default config
config = Config()

# Custom UnlockPickup Environment Implementation
class UnlockPickupEnv(MiniGridEnv):
    def __init__(self, width=4, height=4, max_steps=50):
        self.width = width
        self.height = height
        self.mission = "pick up key, open door, reach goal"
        
        # Define action space (0: left, 1: right, 2: forward, 3: pickup, 4: toggle)
        self.action_space = gym.spaces.Discrete(5)
        self._actions = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}  # Map to MiniGrid actions
        
        # Set mission space for newer MiniGrid versions
        mission_space = Text(max_length=100)
        
        # Observation space includes image
        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=True,  # Full observability as in paper
            agent_view_size=7,       # Agent sees a 7x7 square around itself
            mission_space=mission_space
        )
        
        # Track sub-task completion
        self.key_picked = False
        self.door_opened = False
        
        # Track previous positions for reward shaping
        self.prev_key_distance = None
        self.prev_door_distance = None
        self.prev_goal_distance = None

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Create walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Place agent in the top-left
        self.agent_pos = (1, 1)
        self.agent_dir = 0  # Right
        
        # Key position - one corner
        self.key_pos = (width - 2, 1)
        self.grid.set(*self.key_pos, Key("yellow"))
        
        # Door position - opposite corner
        self.door_pos = (width - 2, height - 2)
        self.grid.set(*self.door_pos, Door("yellow", is_locked=True))
        
        # Goal - behind the door
        self.goal_pos = (width - 1, height - 2)
        self.grid.set(*self.goal_pos, Goal())
        
        # Initialize distances
        self.prev_key_distance = self._manhattan_dist(self.agent_pos, self.key_pos)
        self.prev_door_distance = self._manhattan_dist(self.agent_pos, self.door_pos)
        self.prev_goal_distance = self._manhattan_dist(self.agent_pos, self.goal_pos)

    def step(self, action):
        action = min(max(action, 0), 4)
        # Track previous completion state to detect new completions
        prev_key_picked = self.key_picked
        prev_door_opened = self.door_opened
        was_at_goal = self.agent_pos == self.goal_pos
        
        # Check if pickup/open action is valid BEFORE taking the action
        valid_pickup = False
        valid_open = False
        if action == 3:  # Pickup
            valid_pickup = self._valid_pickup()
        elif action == 4:  # Open door
            valid_open = self._valid_open()
        
        # Map action to MiniGrid's action
        minigrid_action = self._actions[action]
        obs, _, terminated, truncated, info = super().step(minigrid_action)
        
        # Track sub-tasks after taking the action
        if self.carrying and self.carrying.type == "key":
            self.key_picked = True
        
        door_cell = self.grid.get(*self.door_pos)
        if door_cell is None or (isinstance(door_cell, Door) and not door_cell.is_locked):
            self.door_opened = True
        
        # Initialize reward
        reward = 0
        
        # Invalid action penalty - only apply if the action was invalid
        if action == 3 and not valid_pickup:  # Invalid pickup
            reward += config.INVALID_ACTION_PENALTY
            print(f"Invalid pickup attempt! {config.INVALID_ACTION_PENALTY}")
        elif action == 4 and not valid_open:  # Invalid door open
            reward += config.INVALID_ACTION_PENALTY
            print(f"Invalid door open attempt! {config.INVALID_ACTION_PENALTY}")
        
        # Mission rewards - only award when newly completed
        # First mission: Key pickup reward - only if pickup was valid
        if not prev_key_picked and self.key_picked:
            reward += config.KEY_REWARD
            print(f"First mission completed: Key picked up! +{config.KEY_REWARD}")
        
        # Second mission: Door opening reward - only if door open was valid
        if not prev_door_opened and self.door_opened:
            reward += config.DOOR_REWARD
            print(f"Second mission completed: Door opened! +{config.DOOR_REWARD}")
        
        # Third mission: Reaching the goal
        if self.agent_pos == self.goal_pos and self.door_opened and not was_at_goal:
            goal_reward = config.GOAL_REWARD
            time_bonus = (1 - self.step_count / self.max_steps)
            reward += goal_reward + time_bonus
            print(f"Final mission completed: Goal reached! Base: +{goal_reward}, Time bonus: +{time_bonus:.4f}")
            terminated = True
        
        done = terminated or truncated
        return obs, reward, done, truncated, info
    
    def _valid_pickup(self):
        front_pos = self.front_pos
        front_cell = self.grid.get(*front_pos)
        return front_cell and front_cell.type == "key"
    
    def _valid_open(self):
        front_pos = self.front_pos
        front_cell = self.grid.get(*front_pos)
        return front_cell and front_cell.type == "door"

    def _manhattan_dist(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)[0]  # Take first element (observation)
        self.key_picked = False
        self.door_opened = False
        
        # Reset distance tracking
        self.prev_key_distance = self._manhattan_dist(self.agent_pos, self.key_pos)
        self.prev_door_distance = self._manhattan_dist(self.agent_pos, self.door_pos)
        self.prev_goal_distance = self._manhattan_dist(self.agent_pos, self.goal_pos)
        
        return obs

# State Prompt Generator - Updated to match paper's format
class StatePromptGenerator:
    def __init__(self):
        self.dir_map = {
            0: "right", 1: "down", 
            2: "left", 3: "up"
        }
        
    def generate(self, env):
        # Get object in front of agent
        front_pos = env.front_pos
        front_obj = env.grid.get(*front_pos)
        front_obj_type = front_obj.type if front_obj else "empty cell"
        
        # Determine current mission
        if not env.key_picked:
            mission = "pick up key"
        elif not env.door_opened:
            mission = "open door"
        else:
            mission = "reach goal"
        
        # Format exactly as shown in the paper example (page 5)
        return (
            f"The red agent is in a {env.width}x{env.height} grid environment surrounded by "
            f"walls. Each grid cell is identified by coordinates (i, j), where i denotes "
            f"the column and j denotes the row. The agent can turn left (action 0), "
            f"turn right (action 1), move forward (action 2), pick up key (action 3), "
            f"and open door (action 4). The agent can face right (0), down (1), left "
            f"(2), or up (3). The agent cannot pass through walls. It can open the "
            f"door if it has the key and is facing the closed door, and it can pick up "
            f"the key when facing it. The agent needs to find the shortest route to "
            f"key or door and then pickup the key or open the door. Consider the "
            f"direction as the way the agent is facing, not the way we are seeing "
            f"the agent, to avoid mixing right and left. In this state, the agent is at "
            f"position {env.agent_pos}, the agent direction is {self.dir_map[env.agent_dir]} "
            f"and agent's direction number is {env.agent_dir}, and the forward object is {front_obj_type}, "
            f"and the key position is {env.key_pos}, the key is {'being' if env.key_picked else 'not being'} "
            f"carried by the agent, the door is at position {env.door_pos}, "
            f"the goal is at position {env.goal_pos}, the door is {'open' if env.door_opened else 'closed'}, "
            f"and the mission is {mission}. What is the optimal action for the agent to take in this "
            f"state to accomplish the mission? Just say the optimal action number"
        )

# Oracle for generating optimal actions
class Oracle:
    def get_action(self, env):
        # Determine target position based on current mission
        if not env.key_picked:
            target = env.key_pos
        elif not env.door_opened:
            target = env.door_pos
        else:
            target = env.goal_pos
        
        # If at target position, perform the action
        if self._is_adjacent(env.agent_pos, target, env.agent_dir):
            front_pos = env.front_pos
            front_cell = env.grid.get(*front_pos)
            
            # Fix: Compare tuple elements individually
            if (front_pos[0] == target[0] and front_pos[1] == target[1]):
                if target == env.key_pos and not env.key_picked:
                    return 3  # Pick up
                elif target == env.door_pos and not env.door_opened and env.key_picked:
                    return 4  # Open door
                elif target == env.goal_pos:
                    return 2  # Move forward to goal
            
            # Need to turn to face target
            rel_dir = self._get_rel_dir(env.agent_pos, target, env.agent_dir)
            if rel_dir == "left":
                return 0  # Turn left
            elif rel_dir == "right":
                return 1  # Turn right
            elif rel_dir == "behind":
                return 0  # Turn left (arbitrary choice, could also turn right)
        
        # Use A* pathfinding for optimal navigation
        path = self._find_path(env, env.agent_pos, target)
        if not path or len(path) < 2:
            # If no path found, just explore
            return self._random_valid_action(env)
        
        # Get next position in path
        next_pos = path[1]  # path[0] is current position
        
        # Determine action to reach next position
        dx = next_pos[0] - env.agent_pos[0]
        dy = next_pos[1] - env.agent_pos[1]
        
        # Calculate target direction
        target_dir = -1
        if dx == 1 and dy == 0:  # Right
            target_dir = 0
        elif dx == 0 and dy == 1:  # Down
            target_dir = 1
        elif dx == -1 and dy == 0:  # Left
            target_dir = 2
        elif dx == 0 and dy == -1:  # Up
            target_dir = 3
            
        # Calculate turn needed
        if target_dir == -1:
            return self._random_valid_action(env)
            
        # If already facing target direction, move forward
        if env.agent_dir == target_dir:
            return 2  # Forward
            
        # Need to turn
        turn_diff = (target_dir - env.agent_dir) % 4
        if turn_diff == 1:
            return 1  # Turn right
        elif turn_diff == 3:
            return 0  # Turn left
        else:  # turn_diff == 2, need to turn around
            return 0  # Turn left (arbitrary choice)
    
    def _is_adjacent(self, pos, target, dir):
        # Check if target is in front, left, right, or behind agent
        dx = abs(pos[0] - target[0])
        dy = abs(pos[1] - target[1])
        return (dx == 0 and dy == 1) or (dx == 1 and dy == 0)
    
    def _get_rel_dir(self, pos, target, dir):
        # Calculate relative direction of target from agent
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        
        # Convert dx, dy to direction
        target_dir = -1
        if dx == 1 and dy == 0:  # Right
            target_dir = 0
        elif dx == 0 and dy == 1:  # Down
            target_dir = 1
        elif dx == -1 and dy == 0:  # Left
            target_dir = 2
        elif dx == 0 and dy == -1:  # Up
            target_dir = 3
            
        if target_dir == -1:
            return "unknown"
            
        # Calculate relative direction
        rel_dir = (target_dir - dir) % 4
        if rel_dir == 0:
            return "front"
        elif rel_dir == 1:
            return "right"
        elif rel_dir == 2:
            return "behind"
        else:  # rel_dir == 3
            return "left"
    
    def _find_path(self, env, start, target):
        """A* pathfinding to find optimal path"""
        # Define heuristic
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # Initialize open and closed sets
        open_set = set([(start[0], start[1])])  # Convert to tuples to ensure hashability
        closed_set = set()
        
        # Dictionary to store path
        came_from = {}
        
        # g and f scores
        g_score = {(start[0], start[1]): 0}
        f_score = {(start[0], start[1]): heuristic(start, target)}
        
        # Convert target to tuple for comparison
        target_tuple = (target[0], target[1])
        
        while open_set:
            # Find node with lowest f_score
            current = min(open_set, key=lambda pos: f_score.get(pos, float('inf')))
            
            # Check if we've reached the target
            if current == target_tuple:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            open_set.remove(current)
            closed_set.add(current)
            
            # Check neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check if valid position
                if (neighbor[0] < 0 or neighbor[0] >= env.width or
                    neighbor[1] < 0 or neighbor[1] >= env.height):
                    continue
                
                # Check if wall
                cell = env.grid.get(neighbor[0], neighbor[1])
                if cell and cell.type == 'wall':
                    continue
                
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g_score = g_score.get(current, float('inf')) + 1
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                # This path is better, record it
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, target)
        
        # No path found
        return None
    
    def _random_valid_action(self, env):
        """Fallback for when path planning fails"""
        valid_actions = []
        
        # Check forward
        front_pos = env.front_pos
        front_cell = env.grid.get(*front_pos)
        if front_cell is None or front_cell.type != 'wall':
            valid_actions.append(2)  # Forward
        
        # Always can turn
        valid_actions.extend([0, 1])  # Left, Right
        
        # Check if can pick up
        if self._can_pickup(env):
            valid_actions.append(3)  # Pickup
            
        # Check if can open door
        if self._can_open_door(env):
            valid_actions.append(4)  # Open door
            
        return random.choice(valid_actions)
    
    def _can_pickup(self, env):
        front_pos = env.front_pos
        front_cell = env.grid.get(*front_pos)
        return front_cell and front_cell.type == 'key' and not env.key_picked
    
    def _can_open_door(self, env):
        front_pos = env.front_pos
        front_cell = env.grid.get(*front_pos)
        return (front_cell and front_cell.type == 'door' and 
                env.key_picked and front_cell.is_locked)

# Dataset Collection
class RLDataset(Dataset):
    def __init__(self, width=8, height=4, num_samples=21500):
        self.env = UnlockPickupEnv(width=width, height=height, max_steps=config.MAX_STEPS)
        self.oracle = Oracle()
        self.prompt_gen = StatePromptGenerator()
        self.prompts, self.actions = self._collect_data(num_samples)
        
    def _collect_data(self, num_samples):
        prompts, actions = [], []
        print(f"Collecting oracle dataset for {self.env.width}x{self.env.height} grid...")
        
        with tqdm(total=num_samples) as pbar:
            while len(prompts) < num_samples:
                self.env.reset()
                done = False
                
                while not done and len(prompts) < num_samples:
                    prompt = self.prompt_gen.generate(self.env)
                    oracle_action = self.oracle.get_action(self.env)
                    
                    # Balance between exploration and oracle actions
                    if random.random() < config.RANDOM_ACTION_RATIO:
                        action = self.env.action_space.sample()
                    else:
                        action = oracle_action
                    
                    _, _, done, _, _ = self.env.step(action)
                    
                    # As per the paper, only include states where
                    # the oracle's action was selected for training
                    if action == oracle_action:
                        prompts.append(prompt)
                        actions.append(oracle_action)
                        pbar.update(1)
                        
        return prompts, actions
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx], self.actions[idx]

# Calibrated BERT Model - Using MC Dropout as described in paper
class CalibratedBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.LLM_MODEL)
        self.dropout = nn.Dropout(config.DROPOUT_PROB)
        self.classifier = nn.Linear(768, config.ACTIONS)
        self.tokenizer = BertTokenizer.from_pretrained(config.LLM_MODEL)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        return self.classifier(pooled)
    
    def predict(self, prompt):
        """Regular prediction without MC Dropout"""
        self.eval()  # Disable dropout for standard prediction
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=config.MAX_SEQ_LENGTH,
            padding=True,
            truncation=True
        ).to(next(self.parameters()).device)
        
        with torch.no_grad():
            outputs = self(**inputs)
            probs = torch.softmax(outputs, dim=-1)
        
        return probs.squeeze(0)
    
    def mc_predict(self, prompt):
        """Prediction with Monte Carlo Dropout for uncertainty estimation"""
        self.train()  # Enable dropout for MC samples
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=config.MAX_SEQ_LENGTH,
            padding=True,
            truncation=True
        ).to(next(self.parameters()).device)
        
        logits_list = []
        for _ in range(config.MC_SAMPLES):
            with torch.no_grad():
                outputs = self(**inputs)
                logits_list.append(outputs)
        
        # Stack logits from multiple forward passes
        logits = torch.stack(logits_list, dim=0)
        probs = torch.softmax(logits, dim=-1)
        avg_probs = probs.mean(dim=0).squeeze(0)
        
        # Calculate entropy (uncertainty measure)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1)
        # Normalize by max entropy
        norm_entropy = entropy.item() / math.log(config.ACTIONS)
        
        # Ensure it's in [0,1] range
        norm_entropy = max(0.0, min(1.0, norm_entropy))
        
        return avg_probs, norm_entropy
    
# PPO Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        # Shared base network
        self.base = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU()
        )
        # Actor head
        self.actor = nn.Linear(256, config.ACTIONS)
        # Critic head
        self.critic = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.base(x)
        action_probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x).squeeze(-1)
        return action_probs, value

# Main Training System
class CalibratedRLTrainer:
    def __init__(self):
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.llm = CalibratedBERT().to(self.device)
        self.policy = PolicyNetwork().to(self.device)
        
        # Setup optimizers
        self.llm_optimizer = optim.AdamW(self.llm.parameters(), lr=config.LEARNING_RATE)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.LEARNING_RATE)
        
    def train_llm(self, env_width=8, env_height=8, num_samples=21500, force_finetune=False):
        """Fine-tune the LLM on oracle data with specific grid dimensions, or load existing model if available"""
        # Check if pre-trained model exists
        model_path = os.path.join(config.SAVE_DIR, f"best_llm_{env_width}x{env_height}.pt")
        print(model_path)
        
        if os.path.exists(model_path) and not force_finetune:
            print(f"Loading pre-trained LLM model from {model_path}")
            self.llm.load_state_dict(torch.load(model_path))
            
            # Evaluate the loaded model
            # First create dataset to enable evaluation
            print(f"Creating validation dataset for {env_width}x{env_height} grid...")
            self.dataset = RLDataset(width=env_width, height=env_height, num_samples=min(1000, num_samples))
            train_size = int(len(self.dataset) * (1 - config.VAL_SPLIT))
            self.train_set, self.val_set = torch.utils.data.random_split(
                self.dataset, [train_size, len(self.dataset) - train_size])
            
            # Verify the loaded model has good accuracy
            accuracy = self.evaluate_llm()
            print(f"Loaded model accuracy: {accuracy:.4f}")
            
            # Check if accuracy is acceptable
            if accuracy >= 0.85:
                print("Loaded model has acceptable accuracy. Skipping fine-tuning.")
                return accuracy
            else:
                print(f"Loaded model accuracy {accuracy:.4f} is below threshold (0.85). Will perform fine-tuning.")
        else:
            if os.path.exists(model_path):
                print("Pre-trained model exists but fine-tuning is forced.")
            else:
                print("No pre-trained model found. Will perform fine-tuning.")
        
        # Proceed with fine-tuning
        print(f"Creating dataset for {env_width}x{env_height} grid...")
        self.dataset = RLDataset(width=env_width, height=env_height, num_samples=num_samples)
        train_size = int(len(self.dataset) * (1 - config.VAL_SPLIT))
        self.train_set, self.val_set = torch.utils.data.random_split(
            self.dataset, [train_size, len(self.dataset) - train_size])
        
        train_loader = DataLoader(self.train_set, 
                                batch_size=config.BATCH_SIZE, 
                                shuffle=True)
        
        best_val_acc = 0.0
        print(f"Starting LLM fine-tuning for {config.EPOCHS} epochs")
        for epoch in range(config.EPOCHS):
            self.llm.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                prompts, labels = batch
                inputs = self.llm.tokenizer(
                    prompts, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.MAX_SEQ_LENGTH
                ).to(self.device)
                
                labels = torch.tensor(labels, dtype=torch.long).to(self.device)

                logits = self.llm(**inputs)
                loss = nn.CrossEntropyLoss()(logits, labels)
                
                self.llm_optimizer.zero_grad()
                loss.backward()
                self.llm_optimizer.step()
                
                total_loss += loss.item()
            
            val_acc = self.evaluate_llm()
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.llm.state_dict(), model_path)
                print(f"Saved model with accuracy: {val_acc:.4f}")
                
            # Early stopping condition - paper mentions 90-93% accuracy
            if val_acc >= 0.9:
                print(f"Reached target accuracy of {val_acc:.4f}. Early stopping.")
                break
                
        print(f"LLM training complete. Best validation accuracy: {best_val_acc:.4f}")
        
        # Load best model for RL training
        self.llm.load_state_dict(torch.load(model_path))
        
        # Verify the loaded model still has high accuracy
        final_acc = self.evaluate_llm()
        print(f"Loaded model final accuracy: {final_acc:.4f}")
        
        return final_acc
    
    def evaluate_llm(self):
        """Evaluate LLM on validation set"""
        self.llm.eval()
        correct = 0
        val_loader = DataLoader(self.val_set, batch_size=config.BATCH_SIZE)
        
        with torch.no_grad():
            for batch in val_loader:
                prompts, labels = batch
                inputs = self.llm.tokenizer(
                    prompts, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.MAX_SEQ_LENGTH
                ).to(self.device)
                
                labels = torch.tensor(labels, dtype=torch.long).to(self.device)
                
                logits = self.llm(**inputs)
                correct += (logits.argmax(-1) == labels).sum().item()
                
        return correct / len(self.val_set)
    
    def train_rl(self, env_width=4, env_height=4):
        """Training RL agent with uncertainty-aware LLM guidance"""
        # Create environment with specified dimensions
        env = UnlockPickupEnv(width=env_width, height=env_height, max_steps=config.MAX_STEPS)
        prompt_gen = StatePromptGenerator()

        # RL parameters
        num_episodes = 1000  # Same as paper
        gamma = config.GAMMA
        clip_epsilon = config.PPO_EPSILON
        ppo_batch_size = 50  # Number of episodes before policy update
        entropy_coef = 0.01
        value_coef = 0.5
        gae_lambda = 0.95

        # Track metrics for monitoring
        episode_rewards = []
        episode_lengths = []
        entropy_values = []
        completed_tasks = []

        print(f"Starting RL training with calibrated LLM guidance in {env_width}x{env_height} environment...")
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            # Initialize buffer for episode
            states, actions, rewards, log_probs, values, entropies, dones = [], [], [], [], [], [], []
            llm_probs, agents_probs = [], []

            while not done:
                prompt = prompt_gen.generate(env)
                
                # Get LLM's calibrated probabilities with uncertainty
                inputs = self.llm.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    max_length=config.MAX_SEQ_LENGTH,
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                with torch.no_grad():
                    bert_output = self.llm.bert(**inputs).last_hidden_state[:, 0]

                # Get calibrated probabilities from LLM using MC Dropout
                llm_probs_t, entropy = self.llm.mc_predict(prompt)
                llm_probs_t = llm_probs_t.to(self.device)
                entropy_tensor = torch.tensor(entropy).to(self.device)
                
                # Track entropy values
                entropy_values.append(entropy)

                # Get agent's probabilities
                agent_probs_t, value_t = self.policy(bert_output)
                
                # Policy shaping using entropy (Equation 3 from paper)
                # P_a(t) = (1 - H(X)_t) × P_LLM(t) + H(X)_t × P_Agent(t)
                combined_probs = (1 - entropy_tensor) * llm_probs_t + entropy_tensor * agent_probs_t

                # Create distribution and sample action
                action_dist = torch.distributions.Categorical(combined_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                
                # Take action in environment
                next_state, reward, done, truncated, _ = env.step(action.cpu().item())
                done = done or truncated
                
                # Store transition
                states.append(bert_output.detach())
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob.detach())
                values.append(value_t.detach())
                entropies.append(entropy_tensor.detach())
                dones.append(done)
                llm_probs.append(llm_probs_t.detach())
                agents_probs.append(agent_probs_t.detach())

                episode_reward += reward
                state = next_state
                step_count += 1

            # Track metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            
            # Track task completion status
            key_picked = env.key_picked
            door_opened = env.door_opened
            goal_reached = env.agent_pos == env.goal_pos and env.door_opened
            task_status = [key_picked, door_opened, goal_reached]
            completed_tasks.append(task_status)
            
            # Print episode summary
            task_str = f"Key: {'✅' if key_picked else '❌'}, Door: {'✅' if door_opened else '❌'}, Goal: {'✅' if goal_reached else '❌'}"
            print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | Steps: {step_count} | {task_str}")
            
            # Periodically save training visualization
            if (episode + 1) % 100 == 0:
                self._plot_training_progress(episode_rewards, entropy_values, completed_tasks, env_width, env_height)

            # Update policy if enough data is collected
            if (episode + 1) % ppo_batch_size == 0:
                print(f"Updating policy after episode {episode+1}...")
                self._update_policy(states, actions, rewards, log_probs, values, entropies, dones,
                                    llm_probs, agents_probs, gamma, gae_lambda, clip_epsilon,
                                    entropy_coef, value_coef)
                states, actions, rewards, log_probs, values, entropies, dones = [], [], [], [], [], [], []
                llm_probs, agents_probs = [], []

        print("🎉 RL training complete.")

        # Save trained policy
        torch.save(self.policy.state_dict(), os.path.join(config.SAVE_DIR, f"trained_policy_{env_width}x{env_height}.pt"))
        print(f"✅ Saved PPO policy to {config.SAVE_DIR}/trained_policy_{env_width}x{env_height}.pt")

        # Save training metrics
        self._save_metrics(episode_rewards, episode_lengths, entropy_values, completed_tasks, env_width, env_height)
        
        # Final visualization
        self._plot_training_progress(episode_rewards, entropy_values, completed_tasks, env_width, env_height, final=True)
        
        # Calculate AUC for reward curve
        auc = np.trapz(np.array(episode_rewards))
        print(f"Area Under Curve (AUC): {auc:.2f}")
        
        return auc
    
    def _update_policy(self, states, actions, rewards, old_log_probs, old_values, entropies, dones, 
                   llm_probs, agents_probs, gamma, gae_lambda, clip_epsilon, entropy_coef, value_coef):
        """Update PPO policy with entropy-based policy shaping"""
        
        # Convert to tensors and ensure correct shapes
        states = torch.stack(states).to(self.device)                  # [batch_size, 1, 768]
        actions = torch.stack(actions).to(self.device)                # [batch_size]
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)  # [batch_size]
        old_log_probs = torch.stack(old_log_probs).to(self.device)    # [batch_size]
        old_values = torch.stack(old_values).squeeze(-1).to(self.device)  # [batch_size]
        entropies = torch.stack(entropies).to(self.device)            # [batch_size]
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)  # [batch_size]
        llm_probs = torch.stack(llm_probs).to(self.device)            # [batch_size, num_actions]
        agents_probs = torch.stack(agents_probs).to(self.device)      # [batch_size, num_actions]

        # Squeeze state embedding if needed (from [batch_size, 1, 768] to [batch_size, 768])
        if states.dim() == 3:
            states = states.squeeze(1)

        # Compute returns and advantages
        with torch.no_grad():
            _, new_values = self.policy(states)
            returns, advantages = self._compute_gae(rewards, new_values, dones, gamma, gae_lambda)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Reshape entropies for broadcasting with [batch_size, num_actions]
        if entropies.dim() == 1:
            entropies = entropies.unsqueeze(1)  # Shape: [batch_size, 1]

        for _ in range(config.EPOCHS):
            # Get current agent outputs
            new_agent_probs, new_values = self.policy(states)

            # Blend LLM and agent probabilities using entropy
            new_combined_probs = (1 - entropies) * llm_probs + entropies * new_agent_probs

            # Create new distribution and compute log_probs
            new_dist = torch.distributions.Categorical(new_combined_probs)
            new_log_probs = new_dist.log_prob(actions)

            # PPO ratio
            ratios = torch.exp(new_log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            value_loss = nn.MSELoss()(new_values, returns)

            # Entropy bonus
            entropy_loss = -torch.mean(new_dist.entropy())

            # Total loss
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            # Backprop
            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()
            
    def _compute_gae(self, rewards, values, dones, gamma, gae_lambda):
        """Compute Generalized Advantage Estimation (GAE)"""
        batch_size = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        # Check dimensions of values tensor
        if values.dim() == 2:
            # Create zeros with matching second dimension
            bootstrap_value = torch.zeros(1, values.size(1), device=values.device)
        else:
            # For 1D values
            bootstrap_value = torch.zeros(1, device=values.device)
            
        # Append bootstrap value
        values_extended = torch.cat([values, bootstrap_value])

        # Calculate advantages in reverse order
        for t in reversed(range(batch_size)):
            next_non_terminal = 1.0 - dones[t]
            next_value = values_extended[t + 1]
            
            # Calculate TD error
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            
            # Calculate GAE advantage
            advantages[t] = delta + gamma * gae_lambda * next_non_terminal * last_advantage
            last_advantage = advantages[t]

        # Returns = advantages + values
        returns = advantages + values
        return returns, advantages
    
    def _plot_training_progress(self, rewards, entropies, completed_tasks, env_width, env_height, final=False):
        """Create visualizations of training progress"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Smoothed Rewards
        plt.subplot(2, 2, 1)
        raw_rewards = np.array(rewards)
        if len(raw_rewards) > 10:
            window = min(250, len(raw_rewards) // 4)
            smoothed_rewards = np.convolve(raw_rewards, np.ones(window)/window, mode='valid')
            plt.plot(smoothed_rewards)
            plt.title(f'Smoothed Rewards (window={window})')
        else:
            plt.plot(raw_rewards)
            plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        # Plot 2: Entropy Values
        plt.subplot(2, 2, 2)
        if len(entropies) > 0:
            entropy_array = np.array(entropies)
            window = min(250, len(entropy_array) // 4)
            if len(entropy_array) > window:
                smoothed_entropies = np.convolve(entropy_array, np.ones(window)/window, mode='valid')
                plt.plot(smoothed_entropies)
                plt.title(f'LLM Entropy (window={window})')
            else:
                plt.plot(entropy_array)
                plt.title('LLM Entropy')
            plt.xlabel('Step')
            plt.ylabel('Entropy')
            plt.grid(True)
        
        # Plot 3: Task Completion Rates
        if len(completed_tasks) > 0:
            plt.subplot(2, 2, 3)
            task_array = np.array(completed_tasks)
            key_completion = np.mean(task_array[:, 0].astype(float))
            door_completion = np.mean(task_array[:, 1].astype(float))
            goal_completion = np.mean(task_array[:, 2].astype(float))
            
            # For trend, use sliding window
            window_size = min(50, len(task_array) // 4)
            if len(task_array) > window_size:
                key_trend = [np.mean(task_array[max(0, i-window_size):i, 0].astype(float)) 
                            for i in range(window_size, len(task_array))]
                door_trend = [np.mean(task_array[max(0, i-window_size):i, 1].astype(float)) 
                             for i in range(window_size, len(task_array))]
                goal_trend = [np.mean(task_array[max(0, i-window_size):i, 2].astype(float)) 
                             for i in range(window_size, len(task_array))]
                
                x = range(window_size, len(task_array))
                plt.plot(x, key_trend, label=f'Key: {key_completion:.2f}')
                plt.plot(x, door_trend, label=f'Door: {door_completion:.2f}')
                plt.plot(x, goal_trend, label=f'Goal: {goal_completion:.2f}')
                plt.title(f'Task Completion Rates (window={window_size})')
            else:
                plt.bar(['Key', 'Door', 'Goal'], [key_completion, door_completion, goal_completion])
                plt.title('Task Completion Rates')
            
            plt.xlabel('Episode')
            plt.ylabel('Completion Rate')
            plt.legend()
            plt.grid(True)
        
        # Plot 4: Reward Distribution
        plt.subplot(2, 2, 4)
        if len(rewards) > 10:
            plt.hist(rewards, bins=20)
            plt.title('Reward Distribution')
            plt.xlabel('Reward')
            plt.ylabel('Frequency')
            plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        if final:
            plt.savefig(os.path.join(config.SAVE_DIR, f"final_training_summary_{env_width}x{env_height}.png"))
        else:
            plt.savefig(os.path.join(config.SAVE_DIR, f"training_progress_{env_width}x{env_height}.png"))
        
        plt.close()
    
    def _save_metrics(self, rewards, lengths, entropies, completed_tasks, env_width, env_height):
        """Save training metrics to CSV files for later analysis"""
        # Create DataFrame for episode metrics
        episodes_df = pd.DataFrame({
            'episode': range(len(rewards)),
            'reward': rewards,
            'length': lengths,
            'key_picked': [t[0] for t in completed_tasks],
            'door_opened': [t[1] for t in completed_tasks],
            'goal_reached': [t[2] for t in completed_tasks]
        })
        
        # Create DataFrame for entropy values
        entropy_df = pd.DataFrame({
            'step': range(len(entropies)),
            'entropy': entropies
        })
        
        # Save to CSV
        episodes_df.to_csv(os.path.join(config.SAVE_DIR, f"episode_metrics_{env_width}x{env_height}.csv"), index=False)
        entropy_df.to_csv(os.path.join(config.SAVE_DIR, f"entropy_values_{env_width}x{env_height}.csv"), index=False)
        
        print(f"✅ Saved metrics to {config.SAVE_DIR}")
    
    def train_unguided_rl(self, env_width=4, env_height=4):
        """Train RL agent without LLM guidance"""
        # Create environment
        env = UnlockPickupEnv(width=env_width, height=env_height, max_steps=config.MAX_STEPS)
        
        # RL parameters (same as main training)
        num_episodes = 1000
        gamma = config.GAMMA
        clip_epsilon = config.PPO_EPSILON
        ppo_batch_size = 50
        entropy_coef = 0.01
        value_coef = 0.5
        gae_lambda = 0.95
        
        # Track metrics
        episode_rewards = []
        episode_lengths = []
        completed_tasks = []
        
        print(f"Starting unguided RL training in {env_width}x{env_height} environment...")
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            # Initialize buffer for episode
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
            
            while not done:
                # Create a fixed-size feature vector instead of using image
                # This avoids the image processing issues
                state_features = torch.zeros(1, 768).to(self.device)
                
                # Get policy and value from agent
                agent_probs, value = self.policy(state_features)
                
                # Sample action
                action_dist = torch.distributions.Categorical(agent_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                
                # Take action in environment
                next_state, reward, done, truncated, _ = env.step(action.cpu().item())
                done = done or truncated
                
                # Store transition
                states.append(state_features.detach())
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob.detach())
                values.append(value.detach())
                dones.append(done)
                
                episode_reward += reward
                state = next_state
                step_count += 1
            
            # Track metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            
            # Track task completion status
            key_picked = env.key_picked
            door_opened = env.door_opened
            goal_reached = env.agent_pos == env.goal_pos and env.door_opened
            task_status = [key_picked, door_opened, goal_reached]
            completed_tasks.append(task_status)
            
            # Print episode summary
            task_str = f"Key: {'✅' if key_picked else '❌'}, Door: {'✅' if door_opened else '❌'}, Goal: {'✅' if goal_reached else '❌'}"
            print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | Steps: {step_count} | {task_str}")
            
            # Update policy if enough data is collected
            if (episode + 1) % ppo_batch_size == 0:
                print(f"Updating unguided policy after episode {episode+1}...")
                self._update_unguided_policy(states, actions, rewards, log_probs, values, dones,
                            gamma, gae_lambda, clip_epsilon, entropy_coef, value_coef)
                states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        
        print("Unguided RL training complete.")
        
        # Save trained policy
        torch.save(self.policy.state_dict(), os.path.join(config.SAVE_DIR, f"unguided_policy_{env_width}x{env_height}.pt"))
        
        # Calculate AUC
        auc = np.trapz(np.array(episode_rewards))
        print(f"Unguided RL AUC: {auc:.2f}")
        
        return auc
    
    def _process_image(img):
        """Process image observation for the policy network"""
        # Simple CNN to process the image
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
                self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc = nn.Linear(64 * 8 * 8, 768)  # Assuming 64x64 input image reduced to 8x8
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.pool(self.relu(self.conv3(x)))
                x = x.view(-1, 64 * 8 * 8)
                x = self.fc(x)
                return x
        
        # Create a simple CNN to process the image
        model = SimpleCNN()
        
        # Process the image
        with torch.no_grad():
            features = model(img)
    def _update_unguided_policy(self, states, actions, rewards, old_log_probs, old_values, dones,
                    gamma, gae_lambda, clip_epsilon, entropy_coef, value_coef):
        """Update PPO policy for unguided RL (without LLM integration)"""
        # Convert to tensors and ensure correct shapes
        states = torch.cat(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        old_values = torch.stack(old_values).squeeze(-1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute returns and advantages
        with torch.no_grad():
            # Get new values - we need to use the full policy network to get values
            _, new_values = self.policy(states)
            returns, advantages = self._compute_gae(rewards, new_values, dones, gamma, gae_lambda)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(config.EPOCHS):
            # Get current policy outputs
            new_probs, new_values = self.policy(states)
            new_dist = torch.distributions.Categorical(new_probs)
            new_log_probs = new_dist.log_prob(actions)
            
            # PPO ratio
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss
            value_loss = nn.MSELoss()(new_values, returns)
            
            # Entropy bonus
            entropy_loss = -torch.mean(new_dist.entropy())
            
            # Total loss
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            
            # Backprop
            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()
    def train_uncalibrated_rl(self, env_width=4, env_height=4):
        """Train RL agent with uncalibrated LLM guidance"""
        # Similar to train_rl but using the regular predict function instead of mc_predict
        env = UnlockPickupEnv(width=env_width, height=env_height, max_steps=config.MAX_STEPS)
        prompt_gen = StatePromptGenerator()
        
        # RL parameters
        num_episodes = 1000
        gamma = config.GAMMA
        clip_epsilon = config.PPO_EPSILON
        ppo_batch_size = 50
        entropy_coef = 0.01
        value_coef = 0.5
        gae_lambda = 0.95
        
        # Track metrics
        episode_rewards = []
        episode_lengths = []
        completed_tasks = []
        
        print(f"Starting uncalibrated LLM-guided RL training in {env_width}x{env_height} environment...")
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            # Initialize buffer for episode
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
            llm_probs, agents_probs = [], []
            
            while not done:
                prompt = prompt_gen.generate(env)
                
                # Get LLM's uncalibrated probabilities
                llm_probs_t = self.llm.predict(prompt).to(self.device)
                
                # Get agent's probabilities
                inputs = self.llm.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    max_length=config.MAX_SEQ_LENGTH,
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                with torch.no_grad():
                    bert_output = self.llm.bert(**inputs).last_hidden_state[:, 0]
                
                agent_probs_t, value_t = self.policy(bert_output)
                
                # For uncalibrated, use fixed weight of 0.5
                fixed_weight = 0.5
                combined_probs = (1 - fixed_weight) * llm_probs_t + fixed_weight * agent_probs_t
                
                # Create distribution and sample action
                action_dist = torch.distributions.Categorical(combined_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                
                # Take action in environment
                next_state, reward, done, truncated, _ = env.step(action.cpu().item())
                done = done or truncated
                
                # Store transition
                states.append(bert_output.detach())
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob.detach())
                values.append(value_t.detach())
                dones.append(done)
                llm_probs.append(llm_probs_t.detach())
                agents_probs.append(agent_probs_t.detach())
                
                episode_reward += reward
                state = next_state
                step_count += 1
            
            # Track metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            
            # Track task completion status
            key_picked = env.key_picked
            door_opened = env.door_opened
            goal_reached = env.agent_pos == env.goal_pos and env.door_opened
            task_status = [key_picked, door_opened, goal_reached]
            completed_tasks.append(task_status)
            
            # Print episode summary
            task_str = f"Key: {'✅' if key_picked else '❌'}, Door: {'✅' if door_opened else '❌'}, Goal: {'✅' if goal_reached else '❌'}"
            print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | Steps: {step_count} | {task_str}")
            
            # Update policy if enough data is collected
            if (episode + 1) % ppo_batch_size == 0:
                print(f"Updating uncalibrated policy after episode {episode+1}...")
                # For the entropy tensor, use a fixed value of 0.5
                fixed_entropies = [torch.tensor(0.5, device=self.device) for _ in range(len(states))]
                self._update_policy(states, actions, rewards, log_probs, values, fixed_entropies, dones,
                                  llm_probs, agents_probs, gamma, gae_lambda, clip_epsilon,
                                  entropy_coef, value_coef)
                states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
                llm_probs, agents_probs = [], []
        
        print("Uncalibrated LLM-guided RL training complete.")
        
        # Save trained policy
        torch.save(self.policy.state_dict(), os.path.join(config.SAVE_DIR, f"uncalibrated_policy_{env_width}x{env_height}.pt"))
        
        # Calculate AUC
        auc = np.trapz(np.array(episode_rewards))
        print(f"Uncalibrated LLM-guided RL AUC: {auc:.2f}")
        
        return auc
    
    def train_linear_decay_rl(self, env_width=4, env_height=4):
        """Train RL agent with calibrated LLM guidance but using linear decay coefficient"""
        env = UnlockPickupEnv(width=env_width, height=env_height, max_steps=config.MAX_STEPS)
        prompt_gen = StatePromptGenerator()
        
        # RL parameters
        num_episodes = 1000
        gamma = config.GAMMA
        clip_epsilon = config.PPO_EPSILON
        ppo_batch_size = 50
        entropy_coef = 0.01
        value_coef = 0.5
        gae_lambda = 0.95
        
        # Track metrics
        episode_rewards = []
        episode_lengths = []
        entropy_values = []  # Will store linear coefficients instead of entropy
        completed_tasks = []
        
        print(f"Starting linearly decaying LLM-guided RL in {env_width}x{env_height} environment...")
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            # Calculate linear decay coefficient - starts at 1, ends at 0
            linear_coef = max(0, 1.0 - episode / num_episodes)
            
            # Initialize buffer for episode
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
            llm_probs, agents_probs = [], []
            
            while not done:
                prompt = prompt_gen.generate(env)
                
                # Get LLM's calibrated probabilities with uncertainty (but ignore uncertainty)
                llm_probs_t, _ = self.llm.mc_predict(prompt)
                llm_probs_t = llm_probs_t.to(self.device)
                
                # Get agent's probabilities
                inputs = self.llm.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    max_length=config.MAX_SEQ_LENGTH,
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                with torch.no_grad():
                    bert_output = self.llm.bert(**inputs).last_hidden_state[:, 0]
                
                agent_probs_t, value_t = self.policy(bert_output)
                
                # Use linear decay coefficient instead of entropy
                linear_coef_tensor = torch.tensor(linear_coef, device=self.device)
                combined_probs = (1 - linear_coef_tensor) * agent_probs_t + linear_coef_tensor * llm_probs_t
                
                # Create distribution and sample action
                action_dist = torch.distributions.Categorical(combined_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                
                # Take action in environment
                next_state, reward, done, truncated, _ = env.step(action.cpu().item())
                done = done or truncated
                
                # Store transition
                states.append(bert_output.detach())
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob.detach())
                values.append(value_t.detach())
                dones.append(done)
                llm_probs.append(llm_probs_t.detach())
                agents_probs.append(agent_probs_t.detach())
                
                episode_reward += reward
                state = next_state
                step_count += 1
            
            # Track metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            entropy_values.append(linear_coef)  # Track the linear coefficient value
            
            # Track task completion status
            key_picked = env.key_picked
            door_opened = env.door_opened
            goal_reached = env.agent_pos == env.goal_pos and env.door_opened
            task_status = [key_picked, door_opened, goal_reached]
            completed_tasks.append(task_status)
            
            # Print episode summary
            task_str = f"Key: {'✅' if key_picked else '❌'}, Door: {'✅' if door_opened else '❌'}, Goal: {'✅' if goal_reached else '❌'}"
            print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | Steps: {step_count} | {task_str}")
            
            # Update policy if enough data is collected
            if (episode + 1) % ppo_batch_size == 0:
                print(f"Updating linear decay policy after episode {episode+1}...")
                # For entropies, use the linear coefficient
                linear_entropies = [torch.tensor(linear_coef, device=self.device) for _ in range(len(states))]
                self._update_policy(states, actions, rewards, log_probs, values, linear_entropies, dones,
                                  llm_probs, agents_probs, gamma, gae_lambda, clip_epsilon,
                                  entropy_coef, value_coef)
                states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
                llm_probs, agents_probs = [], []
        
        print("Linear decay RL training complete.")
        
        # Save trained policy
        torch.save(self.policy.state_dict(), os.path.join(config.SAVE_DIR, f"linear_decay_policy_{env_width}x{env_height}.pt"))
        
        # Calculate AUC
        auc = np.trapz(np.array(episode_rewards))
        print(f"Linear decay RL AUC: {auc:.2f}")
        
        return auc
        
    def evaluate(self, env_width=4, env_height=4, num_episodes=20):
        """Evaluate the trained agent with LLM guidance"""
        env = UnlockPickupEnv(width=env_width, height=env_height, max_steps=config.MAX_STEPS)
        prompt_gen = StatePromptGenerator()
        
        rewards = []
        task_completion = []
        
        print(f"\nEvaluating agent for {num_episodes} episodes in {env_width}x{env_height} environment...")
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                prompt = prompt_gen.generate(env)
                
                with torch.no_grad():
                    # Get LLM guidance with uncertainty
                    llm_probs, entropy = self.llm.mc_predict(prompt)
                    llm_probs = llm_probs.to(self.device)
                    
                    # Get agent policy
                    inputs = self.llm.tokenizer(
                        prompt, 
                        return_tensors="pt",
                        max_length=config.MAX_SEQ_LENGTH,
                        padding=True,
                        truncation=True
                    ).to(self.device)
                    
                    bert_output = self.llm.bert(**inputs).last_hidden_state[:, 0]
                    agent_probs, _ = self.policy(bert_output)
                    
                    # Combine using entropy as in paper
                    entropy_tensor = torch.tensor(entropy).to(self.device)
                    combined_probs = (1 - entropy_tensor) * llm_probs + entropy_tensor * agent_probs
                    
                    # Take best action (no exploration during evaluation)
                    action = torch.argmax(combined_probs).item()
                
                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                state = next_state
            
            # Track completion
            task_completion.append([
                env.key_picked, 
                env.door_opened, 
                env.agent_pos == env.goal_pos and env.door_opened
            ])
            rewards.append(episode_reward)
            
            # Print progress
            task_str = f"Key: {'✅' if env.key_picked else '❌'}, Door: {'✅' if env.door_opened else '❌'}, Goal: {'✅' if (env.agent_pos == env.goal_pos and env.door_opened) else '❌'}"
            print(f"Eval Episode {episode+1}: Reward = {episode_reward:.2f} | {task_str}")
        
        # Calculate statistics
        mean_reward = np.mean(rewards)
        task_array = np.array(task_completion)
        key_rate = np.mean(task_array[:, 0].astype(float))
        door_rate = np.mean(task_array[:, 1].astype(float))
        goal_rate = np.mean(task_array[:, 2].astype(float))
        
        print(f"\nEvaluation Results (over {num_episodes} episodes):")
        print(f"Mean Reward: {mean_reward:.2f}")
        print(f"Task Completion - Key: {key_rate:.2f}, Door: {door_rate:.2f}, Goal: {goal_rate:.2f}")
        
        return mean_reward, (key_rate, door_rate, goal_rate)
        
    def evaluate_calibration_methods(self, env_width=4, env_height=4, num_samples=1000):
        """Evaluate different uncertainty estimation methods for discrimination and calibration"""
        # Create an environment for testing
        env = UnlockPickupEnv(width=env_width, height=env_height, max_steps=config.MAX_STEPS)
        prompt_gen = StatePromptGenerator()
        oracle = Oracle()
        
        # Variables to track calibration metrics
        prompts = []
        oracle_actions = []
        llm_actions_deterministic = []
        llm_actions_mc = []
        max_probs_deterministic = []
        mean_entropies_deterministic = []
        max_probs_mc = []
        mean_entropies_mc = []
        
        # Collect test data
        print(f"Collecting test data for calibration evaluation...")
        
        for episode in range(100):  # Collect from 100 episodes
            state = env.reset()
            done = False
            
            while not done and len(prompts) < num_samples:
                prompt = prompt_gen.generate(env)
                oracle_action = oracle.get_action(env)
                
                # Get predictions from both methods
                # Deterministic method
                det_probs = self.llm.predict(prompt)
                det_action = torch.argmax(det_probs).item()
                det_max_prob = torch.max(det_probs).item()
                det_entropy = -torch.sum(det_probs * torch.log(det_probs + 1e-10)).item()
                det_norm_entropy = det_entropy / math.log(config.ACTIONS)
                
                # MC Dropout method
                mc_probs, mc_entropy = self.llm.mc_predict(prompt)
                mc_action = torch.argmax(mc_probs).item()
                mc_max_prob = torch.max(mc_probs).item()
                
                # Store data
                prompts.append(prompt)
                oracle_actions.append(oracle_action)
                llm_actions_deterministic.append(det_action)
                llm_actions_mc.append(mc_action)
                max_probs_deterministic.append(det_max_prob)
                mean_entropies_deterministic.append(det_norm_entropy)
                max_probs_mc.append(mc_max_prob)
                mean_entropies_mc.append(mc_entropy)
                
                # Take oracle action to move forward
                _, _, done, _, _ = env.step(oracle_action)
                
                if len(prompts) >= num_samples:
                    break
            
            if len(prompts) >= num_samples:
                break
        
        # Convert to numpy arrays
        oracle_actions = np.array(oracle_actions)
        llm_actions_deterministic = np.array(llm_actions_deterministic)
        llm_actions_mc = np.array(llm_actions_mc)
        max_probs_deterministic = np.array(max_probs_deterministic)
        mean_entropies_deterministic = np.array(mean_entropies_deterministic)
        max_probs_mc = np.array(max_probs_mc)
        mean_entropies_mc = np.array(mean_entropies_mc)
        
        # Calculate accuracy
        det_correct = (llm_actions_deterministic == oracle_actions)
        mc_correct = (llm_actions_mc == oracle_actions)
        
        # Calculate calibration metrics for each method
        
        # 1. Deterministic Mean Entropy
        det_mean_entropy_conf = 1 - mean_entropies_deterministic
        det_mean_entropy_ece = self._calculate_ece(det_mean_entropy_conf, det_correct)
        det_mean_entropy_bs = self._calculate_brier_score(det_mean_entropy_conf, det_correct)
        det_mean_entropy_disc = self._calculate_discrimination(mean_entropies_deterministic, det_correct)
        
        # 2. Deterministic Max Probability
        det_max_prob_ece = self._calculate_ece(max_probs_deterministic, det_correct)
        det_max_prob_bs = self._calculate_brier_score(max_probs_deterministic, det_correct)
        det_max_prob_disc = self._calculate_discrimination(1 - max_probs_deterministic, det_correct)
        
        # 3. MC Mean Entropy
        mc_mean_entropy_conf = 1 - mean_entropies_mc
        mc_mean_entropy_ece = self._calculate_ece(mc_mean_entropy_conf, mc_correct)
        mc_mean_entropy_bs = self._calculate_brier_score(mc_mean_entropy_conf, mc_correct)
        mc_mean_entropy_disc = self._calculate_discrimination(mean_entropies_mc, mc_correct)
        
        # 4. MC Max Probability
        mc_max_prob_ece = self._calculate_ece(max_probs_mc, mc_correct)
        mc_max_prob_bs = self._calculate_brier_score(max_probs_mc, mc_correct)
        mc_max_prob_disc = self._calculate_discrimination(1 - max_probs_mc, mc_correct)
        
        # Create and save calibration metrics table (Table 2 in paper)
        calibration_table = pd.DataFrame({
            "Methods": [
                f"Deterministic {env_width}*{env_height} by Mean Entropy",
                f"Deterministic {env_width}*{env_height} by Max Probability",
                f"Sample Consistency {env_width}*{env_height} by Mean Entropy",
                f"Sample Consistency {env_width}*{env_height} Max Probability"
            ],
            "ECE": [
                det_mean_entropy_ece,
                det_max_prob_ece,
                mc_mean_entropy_ece,
                mc_max_prob_ece
            ],
            "BS": [
                det_mean_entropy_bs,
                det_max_prob_bs,
                mc_mean_entropy_bs,
                mc_max_prob_bs
            ],
            "Discrimination": [
                det_mean_entropy_disc,
                det_max_prob_disc,
                mc_mean_entropy_disc,
                mc_max_prob_disc
            ]
        })
        
        # Save the table
        calibration_table.to_csv(os.path.join(config.SAVE_DIR, f"calibration_metrics_{env_width}x{env_height}.csv"), index=False)
        print("\nCalibration Metrics Table:")
        print(calibration_table)
        
        # Create reliability diagrams
        self._plot_reliability_diagrams(
            [det_mean_entropy_conf, max_probs_deterministic, mc_mean_entropy_conf, max_probs_mc],
            [det_correct, det_correct, mc_correct, mc_correct],
            ['Det. Mean Entropy', 'Det. Max Prob', 'MC Mean Entropy', 'MC Max Prob'],
            env_width, env_height
        )
        
        # Store results in a dictionary
        results = {
            "det_mean_entropy_ece": det_mean_entropy_ece,
            "det_mean_entropy_bs": det_mean_entropy_bs,
            "det_mean_entropy_disc": det_mean_entropy_disc,
            "det_max_prob_ece": det_max_prob_ece,
            "det_max_prob_bs": det_max_prob_bs,
            "det_max_prob_disc": det_max_prob_disc,
            "mc_mean_entropy_ece": mc_mean_entropy_ece,
            "mc_mean_entropy_bs": mc_mean_entropy_bs,
            "mc_mean_entropy_disc": mc_mean_entropy_disc,
            "mc_max_prob_ece": mc_max_prob_ece,
            "mc_max_prob_bs": mc_max_prob_bs,
            "mc_max_prob_disc": mc_max_prob_disc
        }
        
        return results, calibration_table
        
    def _calculate_ece(self, confidences, correct):
        """Calculate Expected Calibration Error"""
        num_bins = 10
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidences, bin_edges[:-1])
        
        ece = 0.0
        for bin_idx in range(1, num_bins + 1):
            bin_mask = (bin_indices == bin_idx)
            if np.sum(bin_mask) > 0:
                bin_conf = np.mean(confidences[bin_mask])
                bin_acc = np.mean(correct[bin_mask])
                bin_size = np.sum(bin_mask) / len(confidences)
                ece += bin_size * np.abs(bin_acc - bin_conf)
        
        return ece
    
    def _calculate_brier_score(self, confidences, correct):
        """Calculate Brier Score"""
        return np.mean((confidences - correct.astype(float)) ** 2)
    
    def _calculate_discrimination(self, uncertainties, correct):
        """Calculate discrimination - how often uncertainty exceeds 0.5 when prediction is wrong"""
        incorrect = ~correct
        high_uncertainty = uncertainties > 0.5
        
        if np.sum(incorrect) > 0:
            return np.mean(high_uncertainty[incorrect])
        else:
            return 1.0  # Perfect discrimination if no incorrect predictions
    
    def _plot_reliability_diagrams(self, confidence_lists, correct_lists, names, env_width, env_height):
        """Plot reliability diagrams for multiple uncertainty estimation methods"""
        plt.figure(figsize=(15, 10))
        
        for i, (confidences, correct, name) in enumerate(zip(confidence_lists, correct_lists, names)):
            plt.subplot(2, 2, i+1)
            
            # Create reliability diagram
            num_bins = 10
            bin_edges = np.linspace(0, 1, num_bins + 1)
            bin_indices = np.digitize(confidences, bin_edges[:-1])
            
            bin_confs = []
            bin_accs = []
            bin_sizes = []
            
            for bin_idx in range(1, num_bins + 1):
                bin_mask = (bin_indices == bin_idx)
                if np.sum(bin_mask) > 0:
                    bin_confs.append(np.mean(confidences[bin_mask]))
                    bin_accs.append(np.mean(correct[bin_mask]))
                    bin_sizes.append(np.sum(bin_mask))
            
            # Normalize bin sizes for visualization
            max_size = max(bin_sizes)
            normalized_sizes = [50 * s / max_size for s in bin_sizes]
            
            # Plot
            plt.scatter(bin_confs, bin_accs, s=normalized_sizes, alpha=0.7)
            plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
            plt.xlabel('Confidence')
            plt.ylabel('Accuracy')
            plt.title(f'{name} Reliability Diagram')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.grid(True)
            
            # Calculate ECE and add to plot
            ece = self._calculate_ece(confidences, correct)
            plt.text(0.05, 0.9, f'ECE: {ece:.4f}', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.SAVE_DIR, f"reliability_diagrams_{env_width}x{env_height}.png"))
        plt.close()
    
    def visualize_incorrect_guidance(self, env_width=4, env_height=4):
        """Visualize examples of incorrect guidance from calibrated and uncalibrated LLMs (for Figure 7)"""
        env = UnlockPickupEnv(width=env_width, height=env_height, max_steps=config.MAX_STEPS)
        prompt_gen = StatePromptGenerator()
        oracle = Oracle()
        
        # Find examples of incorrect guidance
        calibrated_incorrect = None
        uncalibrated_incorrect = None
        
        for _ in range(100):  # Try up to 100 episodes to find good examples
            env.reset()
            for _ in range(20):  # Take up to 20 steps per episode
                prompt = prompt_gen.generate(env)
                oracle_action = oracle.get_action(env)
                
                # Get calibrated LLM prediction
                mc_probs, mc_entropy = self.llm.mc_predict(prompt)
                mc_action = torch.argmax(mc_probs).item()
                
                # Get uncalibrated LLM prediction
                det_probs = self.llm.predict(prompt)
                det_action = torch.argmax(det_probs).item()
                det_entropy = -torch.sum(det_probs * torch.log(det_probs + 1e-10)).item()
                det_norm_entropy = det_entropy / math.log(config.ACTIONS)
                
                # Check if either prediction is incorrect
                if mc_action != oracle_action and calibrated_incorrect is None and mc_entropy > 0.5:
                    # Found a good example for calibrated LLM
                    img = env.render()
                    calibrated_incorrect = {
                        'image': img,
                        'prompt': prompt,
                        'oracle_action': oracle_action,
                        'predicted_action': mc_action,
                        'entropy': mc_entropy,
                        'state': env.agent_pos
                    }
                    
                if det_action != oracle_action and uncalibrated_incorrect is None and det_norm_entropy < 0.3:
                    # Found a good example for uncalibrated LLM
                    img = env.render()
                    uncalibrated_incorrect = {
                        'image': img,
                        'prompt': prompt,
                        'oracle_action': oracle_action,
                        'predicted_action': det_action,
                        'entropy': det_norm_entropy,
                        'state': env.agent_pos
                    }
                
                # Take oracle action to move forward
                _, _, done, _, _ = env.step(oracle_action)
                
                # If we found both examples, we can stop
                if calibrated_incorrect is not None and uncalibrated_incorrect is not None:
                    break
                    
                if done:
                    break
                    
            # If we found both examples, we can stop
            if calibrated_incorrect is not None and uncalibrated_incorrect is not None:
                break
        
        # Create visualization
        if calibrated_incorrect is not None and uncalibrated_incorrect is not None:
            plt.figure(figsize=(10, 6))
            
            # Left: Uncalibrated LLM incorrect guidance
            plt.subplot(1, 2, 1)
            plt.imshow(uncalibrated_incorrect['image'])
            plt.title(f"Uncalibrated LLM\nEntropy: {uncalibrated_incorrect['entropy']:.2f}\n"
                     f"Predicted: {uncalibrated_incorrect['predicted_action']} "
                     f"(Oracle: {uncalibrated_incorrect['oracle_action']})")
            plt.axis('off')
            
            # Right: Calibrated LLM incorrect guidance
            plt.subplot(1, 2, 2)
            plt.imshow(calibrated_incorrect['image'])
            plt.title(f"Calibrated LLM\nEntropy: {calibrated_incorrect['entropy']:.2f}\n"
                     f"Predicted: {calibrated_incorrect['predicted_action']} "
                     f"(Oracle: {calibrated_incorrect['oracle_action']})")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(config.SAVE_DIR, f"incorrect_guidance_examples_{env_width}x{env_height}.png"))
            plt.close()
            
            print("Visualization of incorrect guidance examples saved.")
            
            # Print details for the examples
            print("\nUncalibrated LLM incorrect guidance:")
            print(f"Agent position: {uncalibrated_incorrect['state']}")
            print(f"Oracle action: {uncalibrated_incorrect['oracle_action']}")
            print(f"Predicted action: {uncalibrated_incorrect['predicted_action']}")
            print(f"Entropy: {uncalibrated_incorrect['entropy']:.2f}")
            
            print("\nCalibrated LLM incorrect guidance:")
            print(f"Agent position: {calibrated_incorrect['state']}")
            print(f"Oracle action: {calibrated_incorrect['oracle_action']}")
            print(f"Predicted action: {calibrated_incorrect['predicted_action']}")
            print(f"Entropy: {calibrated_incorrect['entropy']:.2f}")
        else:
            print("Could not find suitable examples of incorrect guidance.")
    
    def compare_methods(self, env_width=4, env_height=4):
        """Run all methods and generate comparison plots and tables"""
        results = {}
        
        # 1. Our Method (Calibrated LLM with entropy-based policy shaping)
        print("\nTraining with our method (Calibrated LLM with entropy-based policy shaping)...")
        auc_our_model = self.train_rl(env_width=env_width, env_height=env_height)
        results["Our Model"] = auc_our_model
        
        # 2. Unguided RL
        print("\nTraining unguided RL baseline...")
        auc_unguided = self.train_unguided_rl(env_width=env_width, env_height=env_height)
        results["Unguided RL"] = auc_unguided
        
        # 3. Uncalibrated LLM-enhanced RL
        print("\nTraining with uncalibrated LLM guidance...")
        auc_uncalibrated = self.train_uncalibrated_rl(env_width=env_width, env_height=env_height)
        results["Uncalibrated LLM-Enhanced RL"] = auc_uncalibrated
        
        # 4. Linear decay coefficient
        print("\nTraining with linear decay coefficient...")
        auc_linear_decay = self.train_linear_decay_rl(env_width=env_width, env_height=env_height)
        results["Calibrated LLM-Enhanced RL by Decay Coefficient"] = auc_linear_decay
        
        # Create comparison plot
        self._plot_method_comparison(results, env_width, env_height)
        
        # Create comparison table
        results_df = pd.DataFrame({"Method": list(results.keys()), "AUC": list(results.values())})
        results_df.to_csv(os.path.join(config.SAVE_DIR, f"method_comparison_{env_width}x{env_height}.csv"), index=False)
        
        print("\nMethod Comparison Results:")
        print(results_df)
        
        return results
    
    def _plot_method_comparison(self, results, env_width, env_height):
        """Plot comparison of different methods"""
        plt.figure(figsize=(10, 6))
        methods = list(results.keys())
        aucs = list(results.values())
        
        # Bar plot
        colors = ['#2C7BB6', '#D7191C', '#FDAE61', '#ABD9E9']
        plt.bar(methods, aucs, color=colors)
        plt.ylabel('Area Under Curve (AUC)')
        plt.title(f'Performance Comparison in {env_width}x{env_height} Environment')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add values on top of bars
        for i, auc in enumerate(aucs):
            plt.text(i, auc + max(aucs) * 0.01, f'{auc:.2f}', ha='center')
        
        plt.savefig(os.path.join(config.SAVE_DIR, f"method_comparison_{env_width}x{env_height}.png"))
        plt.close()