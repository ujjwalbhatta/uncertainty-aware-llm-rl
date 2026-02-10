# dataset.py

from torch.utils.data import Dataset
from tqdm import tqdm
from envs import UnlockPickupEnv
from config import Config

class StatePromptGenerator:
    def generate(self, env):
        prompt = (
            f"The agent is in a {env.width}x{env.height} grid environment surrounded by walls. "
            f"Each grid cell is identified by coordinates (i, j), where i denotes the column "
            f"and j denotes the row. The agent can turn left (action 0), turn right (action 1), "
            f"move forward (action 2), pick up key (action 3), and open door (action 4). "
            f"The agent can face right (0), down (1), left (2), or up (3). "
            f"The agent cannot pass through walls. It can open the door if it has the key and is "
            f"facing the closed door, and it can pick up the key when facing it.\n\n"
            f"Current State:\n"
            f"Agent Position: ({env.agent_pos[0]}, {env.agent_pos[1]})\n"
            f"Agent Direction: {env.agent_dir}\n"
            f"Key Picked: {env.key_picked}\n"
            f"Door Opened: {env.door_opened}\n"
            f"Key Position: ({env.key_pos[0]}, {env.key_pos[1]})\n"
            f"Door Position: ({env.door_pos[0]}, {env.door_pos[1]})\n"
            f"Goal Position: ({env.goal_pos[0]}, {env.goal_pos[1]})\n\n"
            f"Question: What is the optimal action (0-4) for the agent to take?"
        )
        return prompt

class RLDataset(Dataset):
    """
    Dataset for RL Fine-tuning — stores (prompt, oracle_action) pairs.
    """

    def __init__(self, oracle, width=4, height=4, num_samples=21500):
        self.env = UnlockPickupEnv(width=width, height=height, max_steps=Config().MAX_STEPS)
        self.oracle = oracle
        self.prompt_gen = StatePromptGenerator()

        self.prompts, self.actions = self._collect_data(num_samples)

    def _collect_data(self, num_samples):
        prompts, actions = [], []
        print(f"Collecting {num_samples} samples using Oracle...")

        with tqdm(total=num_samples) as pbar:
            while len(prompts) < num_samples:
                self.env.reset()
                done = False

                while not done and len(prompts) < num_samples:
                    # Generate prompt
                    prompt = self.prompt_gen.generate(self.env)

                    # Oracle action
                    oracle_action = self.oracle.get_action(self.env)

                    # Safety remap if action out of bound
                    oracle_action = min(max(oracle_action, 0), 4)

                    # Save prompt and action
                    prompts.append(prompt)
                    actions.append(oracle_action)

                    # Take oracle action
                    _, _, done, _, _ = self.env.step(oracle_action)
                    pbar.update(1)

        return prompts, actions

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.actions[idx]
