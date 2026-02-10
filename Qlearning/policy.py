# policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from config import Config

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=5):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Base network
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Actor and Critic heads
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=Config().PPO_LR)

        # Experience Buffers
        self.reset_buffers()

        # Hyperparameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = Config().PPO_EPSILON
        self.entropy_coef = Config().PPO_ENTROPY_COEF
        self.value_coef = Config().PPO_VALUE_COEF
        self.max_grad_norm = 0.5

        # GRPO KL control
        self.kl_target = 0.01
        self.kl_coef = 0.5

        # Exploration parameters
        self.exploration_rate = 0.3  # Initial exploration rate
        self.exploration_decay = 0.995  # Decay factor

    def forward(self, x):
        base = self.base(x)
        logits = self.actor(base)
        values = self.critic(base)
        probs = F.softmax(logits, dim=-1)
        return probs, values.squeeze(-1)

    def get_action(self, state, llm_probs=None, uncertainty=None):
        if random.random() < self.exploration_rate:
            action = torch.randint(0, 5, (1,)).to(self.device)
            probs, value = self.forward(state)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)
            return action, log_prob, value, dist.entropy(), probs

        probs, value = self.forward(state)

        if llm_probs is not None and uncertainty is not None:
            if isinstance(uncertainty, float):
                uncertainty = torch.tensor(uncertainty).to(state.device)

            llm_weight = 1.0 - uncertainty
            policy_weight = uncertainty

            log_llm = torch.log(llm_probs + 1e-10)
            log_policy = torch.log(probs + 1e-10)
            mixed_log_probs = llm_weight * log_llm + policy_weight * log_policy
            mixed_probs = torch.exp(mixed_log_probs)
            mixed_probs = mixed_probs / (mixed_probs.sum(dim=-1, keepdim=True) + 1e-10)
            dist = torch.distributions.Categorical(mixed_probs)
        else:
            dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value, dist.entropy(), probs

    def store_transition(self, state, action, reward, log_prob, value, done, llm_probs=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        if llm_probs is not None:
            self.llm_probs.append(llm_probs)

    def compute_gae(self, next_value=0):
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        values = torch.tensor(self.values, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(self.device)

        values = torch.cat([values, torch.tensor([next_value]).to(self.device)])
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)

        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.returns = returns

    def update_policy(self, epochs=5, batch_size=64):
        self.compute_gae()

        # Convert buffer data to tensors
        states = torch.cat(self.states)
        actions = torch.tensor(self.actions).to(self.device)
        log_probs_old = torch.tensor(self.log_probs).to(self.device)
        returns = self.returns
        advantages = self.advantages

        using_llm = len(self.llm_probs) > 0
        if using_llm:
            llm_probs = torch.cat(self.llm_probs)

        kl_divergences = []
        losses = []

        for epoch in range(epochs):
            indices = torch.randperm(len(actions))

            for start in range(0, len(actions), batch_size):
                idx = indices[start:start + batch_size]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_log_probs_old = log_probs_old[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]

                policy_probs, values = self.forward(batch_states)
                dist = torch.distributions.Categorical(policy_probs)
                log_probs_new = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                kl_divergence = (torch.exp(batch_log_probs_old) * (batch_log_probs_old - log_probs_new)).mean()
                kl_divergences.append(kl_divergence.item())

                ratio = torch.exp(log_probs_new - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, batch_returns)

                kl_penalty = self.kl_coef * kl_divergence

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy + kl_penalty
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

            mean_kl = sum(kl_divergences) / len(kl_divergences)
            if mean_kl > 2.0 * self.kl_target:
                self.kl_coef *= 1.5
            elif mean_kl < 0.5 * self.kl_target:
                self.kl_coef *= 0.5
            self.kl_coef = max(0.1, min(self.kl_coef, 10.0))

        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(0.01, self.exploration_rate)  # Don't go below 1%

        self.reset_buffers()

        return {
            'policy_loss': sum(losses) / len(losses),
            'final_kl': mean_kl,
            'kl_coef': self.kl_coef,
            'entropy': entropy.item(),
            'exploration_rate': self.exploration_rate
        }

    def reset_buffers(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.llm_probs = []