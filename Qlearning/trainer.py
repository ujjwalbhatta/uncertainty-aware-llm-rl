import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from config import Config
from dataset import RLDataset, StatePromptGenerator
from llm import CalibratedBERT
from policy import PolicyNetwork
from envs import UnlockPickupEnv

class CalibratedRLTrainer:
    def __init__(self):
        self.config = Config()

        self.llm = CalibratedBERT()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = self.llm.to(self.device)

        self.policy = PolicyNetwork().to(self.device)
        self.prompt_gen = StatePromptGenerator()
        self.llm_optimizer = optim.Adam(self.llm.parameters(), lr=self.config.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def train_llm(self, oracle, width=8, height=8, force_finetune=True):
        print("\nStarting LLM Fine-tuning...")
        dataset = RLDataset(oracle, width=width, height=height, num_samples=self.config.DATASET_SIZE)
        val_size = int(self.config.VAL_SPLIT * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE)

        best_val_acc = 0.0

        for epoch in range(1, self.config.EPOCHS + 1):
            start_time = time.time()
            self.llm.train()
            total_loss = 0
            correct = 0
            total = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.EPOCHS}")

            for prompts, actions in progress_bar:
                actions = actions.to(self.device)

                inputs = self.llm.tokenizer(
                    list(prompts),
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.MAX_SEQ_LENGTH
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                logits = self.llm(inputs["input_ids"], inputs["attention_mask"])
                loss = self.criterion(logits, actions)

                self.llm_optimizer.zero_grad()
                loss.backward()
                self.llm_optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == actions).sum().item()
                total += actions.size(0)

                progress_bar.set_postfix({
                    "Batch Loss": f"{loss.item():.4f}",
                    "Train Acc": f"{correct/total:.2%}"
                })

            train_acc = correct / total
            val_acc = self.evaluate_llm(val_loader)
            epoch_time = time.time() - start_time

            print(f"Epoch [{epoch}/{self.config.EPOCHS}] | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | Epoch Time: {epoch_time:.2f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.llm.state_dict(), os.path.join(self.config.SAVE_DIR, "best_llm.pt"))
                print(f"Saved best LLM model with Val Acc: {val_acc:.2%}")

            if best_val_acc >= self.config.TARGET_FINE_TUNE_ACCURACY:
                print(f"Target fine-tuning accuracy {self.config.TARGET_FINE_TUNE_ACCURACY:.2%} reached!")
                break

    def evaluate_llm(self, val_loader):
        self.llm.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for prompts, actions in val_loader:
                actions = actions.to(self.device)

                inputs = self.llm.tokenizer(
                    list(prompts),
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.MAX_SEQ_LENGTH
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                logits = self.llm(inputs["input_ids"], inputs["attention_mask"])
                preds = torch.argmax(logits, dim=-1)

                correct += (preds == actions).sum().item()
                total += actions.size(0)

        return correct / total

    def evaluate_llm(self, val_loader):
        """
        Evaluate LLM on validation dataset.
        """
        self.llm.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for prompts, actions in val_loader:
                actions = actions.to(self.device)

                inputs = self.llm.tokenizer(
                    list(prompts),
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.MAX_SEQ_LENGTH
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                logits = self.llm(inputs["input_ids"], inputs["attention_mask"])
                preds = torch.argmax(logits, dim=-1)

                correct += (preds == actions).sum().item()
                total += actions.size(0)

        return correct / total

    def train_rl(self):
        print("\nStarting RL Training with GRPO...")
        env = UnlockPickupEnv(width=self.config.TRAIN_GRID_WIDTH, height=self.config.TRAIN_GRID_HEIGHT, max_steps=self.config.MAX_STEPS)

        num_episodes = 1000
        max_steps = self.config.MAX_STEPS

        best_avg_reward = float('-inf')
        recent_rewards = []
        stuck_threshold = 10

        for episode in range(1, num_episodes + 1):
            state = env.reset()
            done = False
            episode_reward = 0
            step_count = 0

            prev_state = None
            stuck_steps = 0

            while not done:
                prompt = self.prompt_gen.generate(env)

                llm_probs, uncertainty = self.llm.mc_predict(prompt)
                llm_probs = llm_probs.to(self.device)

                state_emb = self.llm.get_embedding(prompt).unsqueeze(0).to(self.device)

                current_state = (env.agent_pos, env.agent_dir, env.key_picked, env.door_opened)
                if prev_state == current_state:
                    stuck_steps += 1
                else:
                    stuck_steps = 0
                    prev_state = current_state

                if stuck_steps > stuck_threshold:
                    action = torch.randint(0, 5, ()).to(self.device)  # Scalar tensor directly
                    _, log_prob, value, entropy, probs = self.policy.get_action(
                        state_emb, llm_probs=llm_probs, uncertainty=uncertainty
                    )
                else:
                    action, log_prob, value, entropy, probs = self.policy.get_action(
                        state_emb, llm_probs=llm_probs, uncertainty=uncertainty
                    )

                next_state, reward, done, truncated, _ = env.step(action.item())
                done = done or truncated or (step_count >= max_steps - 1)

                self.policy.store_transition(
                    state_emb, action, reward, log_prob, value, done, llm_probs=llm_probs
                )

                episode_reward += reward
                step_count += 1
                state = next_state

            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 50:
                recent_rewards.pop(0)
                avg_reward = sum(recent_rewards) / len(recent_rewards)

                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(self.policy.state_dict(), os.path.join(self.config.SAVE_DIR, "best_policy.pt"))
                    print(f"New best policy saved with avg reward: {best_avg_reward:.2f}")
                elif avg_reward < 0.7 * best_avg_reward and episode > 200:
                    print(f"Performance dropped significantly. Restoring best policy...")
                    self.policy.load_state_dict(torch.load(os.path.join(self.config.SAVE_DIR, "best_policy.pt")))

            print(f"Episode {episode}/{num_episodes} | Reward: {episode_reward:.2f} | Steps: {step_count} | "
                  f"Key: {'✅' if env.key_picked else '❌'}, Door: {'✅' if env.door_opened else '❌'}, "
                  f"Goal: {'✅' if (env.agent_pos == env.goal_pos and env.door_opened) else '❌'} | "
                  f"Exploration: {self.policy.exploration_rate:.2f}")

            if episode % 50 == 0 and len(self.policy.states) > 0:
                print(f"Updating policy at Episode {episode} (batch size: {len(self.policy.states)})...")
                stats = self.policy.update_policy()
                print(f"Policy Loss: {stats['policy_loss']:.4f} | KL: {stats['final_kl']:.6f} | "
                      f"Entropy: {stats['entropy']:.4f} | Exploration: {stats['exploration_rate']:.2f}")

            if episode % self.config.CHECKPOINT_EVERY == 0:
                torch.save(self.policy.state_dict(), os.path.join(self.config.SAVE_DIR, f"policy_checkpoint_{episode}.pt"))
                print(f"Saved policy checkpoint at episode {episode}")

        if len(self.policy.states) > 0:
            print("\n🔄 Final policy update after last episode...")
            stats = self.policy.update_policy()
            print(f"Policy Loss: {stats['policy_loss']:.4f} | KL: {stats['final_kl']:.6f} | Entropy: {stats['entropy']:.4f}")

        torch.save(self.policy.state_dict(), os.path.join(self.config.SAVE_DIR, "final_policy.pt"))
        print(f"RL Training complete! Final policy saved.")

        if os.path.exists(os.path.join(self.config.SAVE_DIR, "best_policy.pt")):
            print(f"Note: For evaluation, consider using 'best_policy.pt' which achieved an average reward of {best_avg_reward:.2f}")
