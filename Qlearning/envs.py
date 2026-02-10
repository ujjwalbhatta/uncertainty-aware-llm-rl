# envs.py

import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Key, Goal
from gymnasium.spaces import Text
from config import Config

class UnlockPickupEnv(MiniGridEnv):
    def __init__(self, width=4, height=4, max_steps=50):
        self.width = width
        self.height = height
        self.mission = "pick up key, open door, reach goal"

        # Define a reduced discrete action space: 5 actions
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

    def _gen_grid(self, width, height):
        """
        Generate a simple grid with:
        - Agent at (1,1)
        - Key at (width-2, 1)
        - Locked door at (width-2, height-2)
        - Goal at (width-1, height-2)
        """
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.agent_pos = (1, 1)
        self.agent_dir = 0  # Facing right initially

        self.key_pos = (width - 2, 1)
        self.grid.set(*self.key_pos, Key("yellow"))

        self.door_pos = (width - 2, height - 2)
        self.grid.set(*self.door_pos, Door("yellow", is_locked=True))

        self.goal_pos = (width - 1, height - 2)
        self.grid.set(*self.goal_pos, Goal())

        self.key_picked = False
        self.door_opened = False

    def step(self, action):
        """
        Take a step in the environment using reduced action space.
        """
        action = min(max(action, 0), 4)
        minigrid_action = self._actions[action]

        obs, reward, terminated, truncated, info = super().step(minigrid_action)

        # Update internal state based on environment after step
        if self.carrying and self.carrying.type == "key":
            self.key_picked = True

        door_cell = self.grid.get(*self.door_pos)
        if door_cell is None or (isinstance(door_cell, Door) and not door_cell.is_locked):
            self.door_opened = True

        config = Config()

        # Reset reward
        reward = 0
    
         # Key pickup reward
        if action == 3:  # Pickup
            if not self.key_picked and self._valid_pickup():
                reward += config.KEY_REWARD
                self.key_picked = True
            elif not self._valid_pickup():
                reward += config.INVALID_ACTION_PENALTY
        
        # Door opening reward
        elif action == 4:  # Toggle (open door)
            if self.key_picked and not self.door_opened and self._valid_open():
                reward += config.DOOR_REWARD
                self.door_opened = True
            elif not self._valid_open():
                reward += config.INVALID_ACTION_PENALTY
        
        # Goal reaching reward 
        if self.agent_pos == self.goal_pos and self.door_opened:
            reward += config.GOAL_REWARD + (1 - self.step_count / self.max_steps)
            terminated = True
        
        done = terminated or truncated
        return obs, reward, done, truncated, info

    def _valid_pickup(self):
        """
        Check if a key is directly in front of the agent.
        """
        front_pos = self.front_pos
        front_cell = self.grid.get(*front_pos)
        return front_cell and front_cell.type == "key"

    def _valid_open(self):
        """
        Check if a door is directly in front of the agent.
        """
        front_pos = self.front_pos
        front_cell = self.grid.get(*front_pos)
        return front_cell and front_cell.type == "door"

    def reset(self, **kwargs):
        """
        Reset environment state at the beginning of an episode.
        """
        obs = super().reset(**kwargs)[0]
        self.key_picked = False
        self.door_opened = False
        return obs
