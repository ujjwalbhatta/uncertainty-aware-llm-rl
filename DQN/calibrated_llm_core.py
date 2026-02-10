import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Key, Goal
from minigrid.minigrid_env import MiniGridEnv
import gymnasium as gym
from gymnasium.spaces import Text
from collections import deque
import matplotlib.pyplot as plt


# Initial Configuration
class Config:
    # Environment 
    TRAIN_GRID_WIDTH = 8
    TRAIN_GRID_HEIGHT = 4
    EVAL_GRID_WIDTH = 4
    EVAL_GRID_HEIGHT = 4
    MAX_STEPS = 50
    ACTIONS = 5  # 0:left, 1:right, 2:forward, 3:pickup, 4:toggle
    
    # LLM Settings
    LLM_MODEL = "bert-base-uncased"
    DROPOUT_PROB = 0.1
    MC_SAMPLES = 8
    
    # Training
    BATCH_SIZE = 16 
    LEARNING_RATE = 5e-5
    PPO_EPSILON = 0.2
    GAMMA = 0.99
    EPOCHS = 5
    MAX_SEQ_LENGTH = 512
    
    # Dataset
    TRAIN_SAMPLES = 21500
    VAL_SPLIT = 0.1
    RANDOM_ACTION_RATIO = 0.3
    
    # Paths
    SAVE_DIR = "calibrated_llm_rl/4x4_results"
    
    # Rewards
    KEY_REWARD = 0.5
    DOOR_REWARD = 0.5
    GOAL_REWARD = 0.2
    INVALID_ACTION_PENALTY = -0.02
    
    # Policy shaping parameters
    ENTROPY_SCALING = 1.0
    
    # DQN Improvements
    DOUBLE_DQN = True
    DUELING_NETWORK = True
    PRIORITIZED_REPLAY = True
    DQN_LEARNING_RATE = 0.0005
    GRPO_LEARNING_RATE = 0.0001
    DQN_BATCH_SIZE = 64
    MEMORY_SIZE = 100000
    TARGET_UPDATE = 1000
    ALPHA = 0.6  # PER: prioritization exponent
    BETA_START = 0.4  # PER: importance sampling start value
    BETA_FRAMES = 100000  # PER: frames to anneal beta to 1.0
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 500

# Initialize default config
config = Config()

# Custom UnlockPickup Environment Implementation
class UnlockPickupEnv(MiniGridEnv):
    def __init__(self, width=4, height=4, max_steps=50):
        self.width = width
        self.height = height
        self.mission = "pick up key, open door, reach goal"
        self.action_space = gym.spaces.Discrete(5)
        self._actions = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}
        
        mission_space = Text(max_length=100)
        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=True,
            agent_view_size=7,
            mission_space=mission_space
        )
        
        self.key_picked = False
        self.door_opened = False
        self.prev_key_distance = None
        self.prev_door_distance = None
        self.prev_goal_distance = None

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        
        self.key_pos = (width - 2, 1)
        self.grid.set(*self.key_pos, Key("yellow"))
        
        self.door_pos = (width - 2, height - 2)
        self.grid.set(*self.door_pos, Door("yellow", is_locked=True))
        
        self.goal_pos = (width - 1, height - 2)
        self.grid.set(*self.goal_pos, Goal())
        
        self.prev_key_distance = self._manhattan_dist(self.agent_pos, self.key_pos)
        self.prev_door_distance = self._manhattan_dist(self.agent_pos, self.door_pos)
        self.prev_goal_distance = self._manhattan_dist(self.agent_pos, self.goal_pos)

    def step(self, action):
        action = min(max(action, 0), 4)
        minigrid_action = self._actions[action]
        obs, reward, terminated, truncated, info = super().step(minigrid_action)
        
        if self.carrying and self.carrying.type == "key":
            self.key_picked = True
        
        door_cell = self.grid.get(*self.door_pos)
        if door_cell is None or (isinstance(door_cell, Door) and not door_cell.is_locked):
            self.door_opened = True
        
        reward = 0
        
        if action == 3:  # Pickup
            if not self.key_picked and self._valid_pickup():
                reward += config.KEY_REWARD
                self.key_picked = True
            elif not self._valid_pickup():
                reward += config.INVALID_ACTION_PENALTY
        
        elif action == 4:  # Toggle
            if self.key_picked and not self.door_opened and self._valid_open():
                reward += config.DOOR_REWARD
                self.door_opened = True
            elif not self._valid_open():
                reward += config.INVALID_ACTION_PENALTY
        
        if self.agent_pos == self.goal_pos and self.door_opened:
            reward += config.GOAL_REWARD + (1 - self.step_count / self.max_steps)
            terminated = True

        done = terminated or truncated

        # Reward shaping
        new_key_distance = self._manhattan_dist(self.agent_pos, self.key_pos)
        if new_key_distance < self.prev_key_distance:
            reward += 0.02

        new_door_distance = self._manhattan_dist(self.agent_pos, self.door_pos)
        if new_door_distance < self.prev_door_distance:
            reward += 0.02

        new_goal_distance = self._manhattan_dist(self.agent_pos, self.goal_pos)
        if new_goal_distance < self.prev_goal_distance:
            reward += 0.02

        self.prev_key_distance = new_key_distance
        self.prev_door_distance = new_door_distance
        self.prev_goal_distance = new_goal_distance

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
        obs = super().reset(**kwargs)[0]
        self.key_picked = False
        self.door_opened = False
        
        self.prev_key_distance = self._manhattan_dist(self.agent_pos, self.key_pos)
        self.prev_door_distance = self._manhattan_dist(self.agent_pos, self.door_pos)
        self.prev_goal_distance = self._manhattan_dist(self.agent_pos, self.goal_pos)
        
        return obs
        
    def get_state_vector(self):
        agent_x = self.agent_pos[0] / self.width
        agent_y = self.agent_pos[1] / self.height
        agent_dir_sin = np.sin(self.agent_dir * np.pi/2)
        agent_dir_cos = np.cos(self.agent_dir * np.pi/2)
        has_key = 1.0 if self.key_picked else 0.0
        door_open = 1.0 if self.door_opened else 0.0
        key_rel_x = (self.key_pos[0] - self.agent_pos[0]) / self.width
        key_rel_y = (self.key_pos[1] - self.agent_pos[1]) / self.height
        door_rel_x = (self.door_pos[0] - self.agent_pos[0]) / self.width
        door_rel_y = (self.door_pos[1] - self.agent_pos[1]) / self.height
        goal_rel_x = (self.goal_pos[0] - self.agent_pos[0]) / self.width
        goal_rel_y = (self.goal_pos[1] - self.agent_pos[1]) / self.height

        state = np.array([
            agent_x, agent_y, 
            agent_dir_sin, agent_dir_cos, 
            has_key, door_open,
            key_rel_x, key_rel_y, 
            door_rel_x, door_rel_y, 
            goal_rel_x, goal_rel_y
        ], dtype=np.float32)
        
        return state
    
# Prioritized Experience Replay Memory
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        """Save a transition to memory"""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, beta=0.4):
        """Sample a batch of transitions based on priorities"""
        if self.size == 0:
            return [], [], []
        
        # Sample based on priorities raised to power alpha
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Get sampled transitions
        samples = [self.memory[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights
        
        return samples, indices, np.array(weights, dtype=np.float32)
    
    def update_priorities(self, indices, priorities):
        """Update priorities after learning"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return self.size

# Dueling DQN Network
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        
        # Feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        features = self.feature_layer(x)
        
        # Calculate value and advantage
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        qvals = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return qvals

# Standard DQN Network (for comparison)
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# Improved DQN Agent (Double DQN + Dueling + Prioritized Replay)
class ImprovedDQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.0005,
        batch_size=64,
        memory_size=100000,
        gamma=0.99,
        target_update=1000,
        double_dqn=True,
        dueling_network=True,
        prioritized_replay=True,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
        device=None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.double_dqn = double_dqn
        self.beta = beta_start
        self.beta_start = beta_start
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up networks based on configuration
        if dueling_network:
            self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
            self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        else:
            self.policy_net = DQN(state_dim, action_dim).to(self.device)
            self.target_net = DQN(state_dim, action_dim).to(self.device)
        
        # Copy weights to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Set up memory type based on configuration
        if prioritized_replay:
            self.memory = PrioritizedReplayMemory(memory_size, alpha)
            self.beta = beta_start
            self.beta_frames = beta_frames
        else:
            self.memory = deque(maxlen=memory_size)
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Training variables
        self.learn_step_counter = 0
        self.beta_step = 0
        self.prioritized_replay = prioritized_replay
    
    def select_action(self, state, epsilon):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay memory"""
        if self.prioritized_replay:
            self.memory.push(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))
    
    def optimize_model(self):
        """Update model weights with a batch of experiences"""
        if self.prioritized_replay:
            if len(self.memory) < 2 * self.batch_size:
                return 0
                
            # Update beta for importance sampling
            self.beta = min(1.0, self.beta + (1.0 - self.beta_start) / self.beta_frames)
            self.beta_step += 1
            
            # Sample batch with priorities
            batch, indices, weights = self.memory.sample(self.batch_size, self.beta)
            
            # Unpack batch
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1).to(self.device)
            weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
            
            # Double DQN: Select action using policy network, evaluate using target network
            if self.double_dqn:
                # Get actions from policy network
                next_q_values = self.policy_net(next_states)
                next_actions = torch.argmax(next_q_values, dim=1, keepdim=True)
                
                # Evaluate actions using target network
                next_q_values_target = self.target_net(next_states)
                next_q_values = next_q_values_target.gather(1, next_actions)
            else:
                # Standard DQN: Evaluate actions using target network only
                next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            
            # Calculate expected Q values
            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Get current Q values
            q_values = self.policy_net(states).gather(1, actions)
            
            # Calculate loss with importance sampling weights
            td_errors = torch.abs(expected_q_values - q_values).detach().cpu().numpy()
            loss = (weights * nn.MSELoss(reduction='none')(q_values, expected_q_values)).mean()
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            
            # Update priorities in memory
            self.memory.update_priorities(indices, td_errors + 1e-5)  # Small constant to avoid zero priority
            
            # Update target network
            self.learn_step_counter += 1
            if self.learn_step_counter % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            return loss.item()
            
        else:
            # Regular experience replay
            if len(self.memory) < 2 * self.batch_size:
                return 0
                
            # Sample random batch
            batch = random.sample(self.memory, self.batch_size)
            
            # Unpack batch
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1).to(self.device)
            
            # Double DQN: Select action using policy network, evaluate using target network
            if self.double_dqn:
                # Get actions from policy network
                next_q_values = self.policy_net(next_states)
                next_actions = torch.argmax(next_q_values, dim=1, keepdim=True)
                
                # Evaluate actions using target network
                next_q_values_target = self.target_net(next_states)
                next_q_values = next_q_values_target.gather(1, next_actions)
            else:
                # Standard DQN: Evaluate actions using target network only
                next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            
            # Calculate expected Q values
            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Get current Q values
            q_values = self.policy_net(states).gather(1, actions)
            
            # Calculate loss
            loss = nn.MSELoss()(q_values, expected_q_values)
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            
            # Update target network
            self.learn_step_counter += 1
            if self.learn_step_counter % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            return loss.item()
    
    def get_q_values(self, state):
        """Get Q-values for a given state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state_tensor).cpu().numpy()[0]



def compute_ece(pred_probs, true_labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (pred_probs >= bins[i]) & (pred_probs < bins[i+1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_confidence = np.mean(pred_probs[bin_mask])
            bin_accuracy = np.mean(true_labels[bin_mask])
            ece += (bin_size / len(pred_probs)) * np.abs(bin_confidence - bin_accuracy)
    return ece

def compute_brier_score(pred_probs, true_labels):
    return np.mean((pred_probs - true_labels) ** 2)

def compute_discrimination(pred_probs, true_labels):
    correct_probs = pred_probs[true_labels == 1]
    incorrect_probs = pred_probs[true_labels == 0]
    if len(correct_probs) == 0 or len(incorrect_probs) == 0:
        return 0.0
    return np.mean(correct_probs) - np.mean(incorrect_probs)

# GRPO Policy Network
class GRPOPolicy(nn.Module):
    def __init__(self, dqn_agent, lr=0.0001, env_width=4, env_height=4, max_steps=50, device=None):
        super(GRPOPolicy, self).__init__()
        self.dqn_agent = dqn_agent
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy network
        self.policy_network = nn.Sequential(
            nn.Linear(dqn_agent.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, dqn_agent.action_dim)
        ).to(self.device)
        
        # Value network
        self.value_network = nn.Sequential(
            nn.Linear(dqn_agent.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)
        
        # Environment
        self.env = UnlockPickupEnv(width=env_width, height=env_height, max_steps=max_steps)
        
        # PPO settings
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.02
        self.epochs = 4
        
        # Exploration settings
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon = self.epsilon_start

        # Metrics logging
        self.logged_probs = []
        self.logged_correct = []
        self.logged_steps = []
        self.global_step_count = 0

    def get_action_probs(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.policy_network(state_tensor)
        return torch.softmax(logits, dim=1)

    def get_value(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.value_network(state_tensor)

    def select_action(self, state):
        dqn_q_values = self.dqn_agent.get_q_values(state)
        
        with torch.no_grad():
            policy_probs = self.get_action_probs(state).cpu().numpy()[0]
        
        combined_probs = np.zeros(self.dqn_agent.action_dim)
        
        if np.max(dqn_q_values) != np.min(dqn_q_values):
            norm_q_values = (dqn_q_values - np.min(dqn_q_values)) / (np.max(dqn_q_values) - np.min(dqn_q_values))
        else:
            norm_q_values = np.ones_like(dqn_q_values) / len(dqn_q_values)
        
        exp_q = np.exp(norm_q_values - np.max(norm_q_values))
        dqn_probs = exp_q / np.sum(exp_q)
        
        alpha = 0.5
        combined_probs = alpha * dqn_probs + (1 - alpha) * policy_probs
        
        if random.random() < self.epsilon:
            return random.randrange(self.dqn_agent.action_dim)
        else:
            return np.argmax(combined_probs)

    def train(self, num_episodes=10000):
        rewards_history = []
        task_success_history = []

        for episode in range(num_episodes):
            state = self.env.reset()
            state_vector = self.env.get_state_vector()
            done = False

            states = []
            actions = []
            rewards = []
            values = []
            dones = []

            episode_reward = 0
            key_picked = False
            door_opened = False
            goal_reached = False

            while not done:
                with torch.no_grad():
                    value = self.get_value(state_vector).item()

                action = self.select_action(state_vector)

                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state_vector = self.env.get_state_vector()

                # Log for calibration metrics
                with torch.no_grad():
                    action_probs = self.get_action_probs(state_vector).cpu().numpy()[0]
                self.logged_probs.append(action_probs[action])
                self.logged_correct.append(1 if goal_reached else 0)
                self.logged_steps.append(self.global_step_count)
                self.global_step_count += 1

                shaped_reward = reward

                states.append(state_vector)
                actions.append(action)
                rewards.append(shaped_reward)
                values.append(value)
                dones.append(done or truncated)

                self.dqn_agent.store_transition(state_vector, action, shaped_reward, next_state_vector, done or truncated)

                state = next_state
                state_vector = next_state_vector

                episode_reward += reward

                self.dqn_agent.optimize_model()

                key_picked = key_picked or self.env.key_picked
                door_opened = door_opened or self.env.door_opened
                goal_reached = goal_reached or (self.env.agent_pos == self.env.goal_pos and self.env.door_opened)

            if len(states) > 0:
                returns, advantages = self.compute_gae(rewards, values, dones)
                self.update_policy(states, actions, rewards, values, dones, returns, advantages)

            self.epsilon = max(self.epsilon_end, self.epsilon * 0.999)

            rewards_history.append(episode_reward)
            task_success_history.append([key_picked, door_opened, goal_reached])

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | Avg(10): {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | Key: {'✅' if key_picked else '❌'} | Door: {'✅' if door_opened else '❌'} | Goal: {'✅' if goal_reached else '❌'}")

            if (episode + 1) % 100 == 0:
                self._plot_progress(rewards_history, task_success_history, episode+1)

        self._plot_progress(rewards_history, task_success_history, num_episodes, final=True)
        self._save_models()

        return rewards_history, task_success_history

    def update_policy(self, states, actions, rewards, values, dones, returns, advantages):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with torch.no_grad():
            old_logits = self.policy_network(states)
            old_probs = torch.softmax(old_logits, dim=1)
            old_log_probs = torch.log(old_probs.gather(1, actions.unsqueeze(1)) + 1e-10).squeeze(1)

        for _ in range(self.epochs):
            logits = self.policy_network(states)
            probs = torch.softmax(logits, dim=1)
            log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)) + 1e-10).squeeze(1)

            values_new = self.value_network(states).squeeze(1)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.MSELoss()(values_new, returns)

            entropy = -(probs * torch.log(probs + 1e-10)).sum(1).mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
            self.policy_optimizer.step()
            self.value_optimizer.step()

    def compute_gae(self, rewards, values, dones):
        gae = 0
        returns = []
        advantages = []

        values.append(0)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)

        return returns, advantages

    def _save_models(self):
        torch.save(self.dqn_agent.policy_net.state_dict(), os.path.join(config.SAVE_DIR, "dqn_improved_model.pt"))
        torch.save(self.policy_network.state_dict(), os.path.join(config.SAVE_DIR, "grpo_policy_model.pt"))
        torch.save(self.value_network.state_dict(), os.path.join(config.SAVE_DIR, "grpo_value_model.pt"))
        print(f"Models saved to {config.SAVE_DIR}")

    def _plot_progress(self, rewards, task_success, episode, final=False):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        plt.subplot(2, 2, 2)
        window = min(100, len(rewards) // 4)
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(smoothed)
            plt.title(f'Smoothed Rewards (window={window})')
            plt.xlabel('Episode')
            plt.ylabel('Reward')

        plt.subplot(2, 2, 3)
        task_array = np.array(task_success)
        if len(task_array) > 0:
            window_size = min(50, len(task_array) // 4)
            if len(task_array) > window_size:
                key_trend = [np.mean(task_array[max(0, i-window_size):i, 0].astype(float)) for i in range(window_size, len(task_array))]
                door_trend = [np.mean(task_array[max(0, i-window_size):i, 1].astype(float)) for i in range(window_size, len(task_array))]
                goal_trend = [np.mean(task_array[max(0, i-window_size):i, 2].astype(float)) for i in range(window_size, len(task_array))]
                x = range(window_size, len(task_array))
                plt.plot(x, key_trend, label='Key')
                plt.plot(x, door_trend, label='Door')
                plt.plot(x, goal_trend, label='Goal')
                plt.title('Task Completion Rates')
                plt.xlabel('Episode')
                plt.ylabel('Success Rate')
                plt.legend()

        plt.subplot(2, 2, 4)
        plt.hist(rewards[-100:] if len(rewards) > 100 else rewards, bins=20)
        plt.title('Recent Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')

        plt.tight_layout()
        if final:
            plt.savefig(os.path.join(config.SAVE_DIR, "dqn_grpo_improved_final.png"))
        else:
            plt.savefig(os.path.join(config.SAVE_DIR, f"dqn_grpo_improved_ep{episode}.png"))
        plt.close()

        if len(self.logged_probs) > 0:
            pred_probs = np.array(self.logged_probs)
            true_labels = np.array(self.logged_correct)
            ece_list, brier_list, disc_list = [], [], []
            window_size = 500

            for i in range(0, len(pred_probs), window_size):
                window_probs = pred_probs[i:i+window_size]
                window_labels = true_labels[i:i+window_size]
                if len(window_probs) > 0:
                    ece_list.append(compute_ece(window_probs, window_labels))
                    brier_list.append(compute_brier_score(window_probs, window_labels))
                    disc_list.append(compute_discrimination(window_probs, window_labels))

            plt.figure(figsize=(12, 6))
            plt.plot(ece_list, label='ECE')
            plt.plot(brier_list, label='Brier Score')
            plt.plot(disc_list, label='Discrimination')
            plt.title('Calibration Metrics Over Training')
            plt.xlabel('Training Step Window')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.grid()

            if final:
                plt.savefig(os.path.join(config.SAVE_DIR, "calibration_metrics_final.png"))
            else:
                plt.savefig(os.path.join(config.SAVE_DIR, f"calibration_metrics_ep{episode}.png"))
            plt.close()
