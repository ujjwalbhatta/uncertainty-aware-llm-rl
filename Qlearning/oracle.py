import random
import numpy as np
import pickle
from collections import defaultdict
from config import Config

class QlearningOracle:
    def __init__(self, env_width=4, env_height=4):
        self.env_width = env_width
        self.env_height = env_height

        # Q-learning hyperparameters
        self.alpha = Config().ALPHA
        self.gamma = Config().GAMMA
        self.epsilon = 1.0  # Starting with full exploration
        self.epsilon_decay = Config().EPSILON_DECAY
        self.min_epsilon = Config().MIN_EPSILON

        # Q-table: maps (state) → (action values)
        self.q_table = defaultdict(lambda: defaultdict(float))

        # Track visited states (for coverage monitoring)
        self.visited_states = set()

    def _get_state_key(self, env):
        return (
            env.agent_pos,
            env.agent_dir,
            env.key_picked,
            env.door_opened
        )

    def get_action(self, env):
        state = self._get_state_key(env)
        self.visited_states.add(state)

        # Explore with probability epsilon
        if random.random() < self.epsilon:
            return random.randint(0, Config().ACTIONS - 1)

        # Exploit best known action
        q_values = self.q_table[state]
        if not q_values:
            return random.randint(0, Config().ACTIONS - 1)

        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def update_q_values(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state].values(), default=0)

        target = reward + (self.gamma * next_max_q * (1 - int(done)))
        new_q = current_q + self.alpha * (target - current_q)

        self.q_table[state][action] = new_q

    def train_episode(self, env, use_shaping=False):  # Default changed to False
        env.reset()  # Ensure clean environment state
        state = self._get_state_key(env)
        done = False
        
        # Initialize episode metrics
        key_reward = 0
        door_reward = 0
        goal_reward = 0
        penalty_reward = 0
        shaping_reward = 0
        step_count = 0
        
        # Check initial state
        key_picked_init = env.key_picked
        door_opened_init = env.door_opened
        goal_reached_init = env.agent_pos == env.goal_pos and env.door_opened

        while not done and step_count < env.max_steps:
            action = self.get_action(env)
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = self._get_state_key(env)
            step_count += 1
            
            # Explicitly track reward sources
            if env.key_picked and not state[2]:  # Key was just picked up
                key_reward += Config().KEY_REWARD
            elif env.door_opened and not state[3]:  # Door was just opened
                door_reward += Config().DOOR_REWARD
            elif env.agent_pos == env.goal_pos and env.door_opened and not goal_reached_init:
                goal_reward += Config().GOAL_REWARD
            
            # Check for penalty
            if reward < 0:
                penalty_reward += reward
            
            # REMOVED: Shaping rewards removed completely

            # Update Q-values
            self.update_q_values(state, action, reward, next_state, done)
            state = next_state

        # Calculate total reward using explicit components
        total_reward = key_reward + door_reward + goal_reward + penalty_reward
        
        # Add time-based bonus for fast completion, as in the original env
        # After (Correct, Fixed)
        if env.agent_pos == env.goal_pos and env.door_opened:
            pass  # Do nothing extra, no time bonus

        
        # Check task completion
        key_picked = env.key_picked
        door_opened = env.door_opened
        goal_reached = env.agent_pos == env.goal_pos and env.door_opened
        
        # Epsilon decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # Occasionally print detailed reward breakdown (5% chance)
        if random.random() < 0.05:
            task_str = f"Key: {'✅' if key_picked else '❌'}, Door: {'✅' if door_opened else '❌'}, Goal: {'✅' if goal_reached else '❌'}"
            print(f"Reward Breakdown: Key: {key_reward:.2f}, Door: {door_reward:.2f}, "
                  f"Goal: {goal_reward:.2f}, Penalties: {penalty_reward:.2f}, "
                  f"Total: {total_reward:.2f} | {task_str}")

        return total_reward

    def train(self, env, num_episodes=1500, use_shaping=False):  # Default changed to False
        """
        Train Q-learning for multiple episodes.
        """
        print(f"🧠 Training Q-learning Oracle for {num_episodes} episodes...")
        rewards = []
        task_completions = {"key": 0, "door": 0, "goal": 0}

        for episode in range(1, num_episodes + 1):
            env.reset()
            total_reward = self.train_episode(env, use_shaping)
            rewards.append(total_reward)
            
            # Track task completion
            if env.key_picked:
                task_completions["key"] += 1
            if env.door_opened:
                task_completions["door"] += 1
            if env.agent_pos == env.goal_pos and env.door_opened:
                task_completions["goal"] += 1

            if episode % 200 == 0:
                avg_reward = np.mean(rewards[-200:])
                key_rate = task_completions["key"] / min(200, episode) * 100
                door_rate = task_completions["door"] / min(200, episode) * 100
                goal_rate = task_completions["goal"] / min(200, episode) * 100
                
                print(f"Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | Key: {key_rate:.1f}% | "
                      f"Door: {door_rate:.1f}% | Goal: {goal_rate:.1f}%")
                
                # Reset task completion counters
                task_completions = {"key": 0, "door": 0, "goal": 0}

        coverage = len(self.visited_states) / (self.env_width * self.env_height * 4 * 2 * 2)
        print(f"✅ Oracle Training Complete. Visited {len(self.visited_states)} unique states ({coverage:.2%} theoretical coverage).")

    def save(self, filepath):
        """
        Save Q-table and parameters to a file.
        """
        q_table_dict = {state: dict(actions) for state, actions in self.q_table.items()}

        data = {
            "q_table": q_table_dict,
            "visited_states": list(self.visited_states),
            "epsilon": self.epsilon,
            "env_width": self.env_width,
            "env_height": self.env_height,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"💾 Saved Oracle model to {filepath}")

    def load(self, filepath):
        """
        Load Q-table and parameters from a file.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        q_table_dict = data["q_table"]
        self.visited_states = set(data["visited_states"])
        self.epsilon = data["epsilon"]
        self.env_width = data["env_width"]
        self.env_height = data["env_height"]

        self.q_table = defaultdict(lambda: defaultdict(float))
        for state, actions in q_table_dict.items():
            for action, value in actions.items():
                self.q_table[state][action] = value

        print(f"📦 Loaded Oracle model from {filepath}")
        print(f"Visited States: {len(self.visited_states)}")