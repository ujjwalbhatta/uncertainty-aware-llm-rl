# llm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig
from config import Config

class CalibratedBERT(nn.Module):
    def __init__(self):
        super().__init__()

        config = Config()

        # Load BERT with custom dropout configuration
        bert_config = BertConfig.from_pretrained(config.LLM_MODEL)
        bert_config.hidden_dropout_prob = config.DROPOUT_PROB
        bert_config.attention_probs_dropout_prob = config.DROPOUT_PROB

        self.bert = BertModel.from_pretrained(config.LLM_MODEL, config=bert_config)

        # Dropout before classification
        self.dropout = nn.Dropout(config.DROPOUT_PROB)

        # Classification head: output logits for 5 actions
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.ACTIONS)

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config.LLM_MODEL)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through BERT + classifier.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Take [CLS] token output
        cls_output = outputs.last_hidden_state[:, 0]  # (batch_size, hidden_dim)

        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return logits

    def predict(self, prompt):
        self.eval()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=Config().MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True
        ).to(next(self.parameters()).device)

        with torch.no_grad():
            logits = self.forward(inputs.input_ids, inputs.attention_mask)
            probs = F.softmax(logits, dim=-1)

        return probs.squeeze(0)  # (5,)

    def mc_predict(self, prompt):
        self.train()  # Enable dropout

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=Config().MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True
        ).to(next(self.parameters()).device)

        logits_list = []
        for _ in range(Config().MC_SAMPLES):
            with torch.no_grad():
                logits = self.forward(inputs.input_ids, inputs.attention_mask)
                logits_list.append(logits)

        logits = torch.stack(logits_list)  # (MC_SAMPLES, batch_size, num_classes)
        probs = F.softmax(logits, dim=-1)

        avg_probs = probs.mean(dim=0).squeeze(0)  # (5,)

        # Calculate entropy
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        max_entropy = torch.log(torch.tensor(Config().ACTIONS, dtype=torch.float32))
        normalized_entropy = entropy / max_entropy

        # Apply sigmoid smoothing
        scaled_entropy = torch.sigmoid(4 * (normalized_entropy - 0.5)).item()

        return avg_probs, scaled_entropy

    def get_embedding(self, prompt):
        """
        Get BERT [CLS] embedding of a prompt.
        """
        self.eval()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=Config().MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True
        ).to(next(self.parameters()).device)

        with torch.no_grad():
            outputs = self.bert(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            embeddings = outputs.last_hidden_state[:, 0]  # (batch_size, hidden_dim)

        return embeddings.squeeze(0)  # (hidden_dim,)
