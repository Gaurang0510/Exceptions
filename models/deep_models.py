"""
Deep Learning Models
=====================
LSTM and Transformer (FinBERT-based) classifiers for financial news
impact prediction.

Models
------
- **LSTMClassifier** — captures sequential patterns in price / text
  token sequences.
- **FinBERTClassifier** — fine-tunes a pre-trained FinBERT transformer
  for 3-class stock impact prediction.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config.settings import (
    MAX_SEQUENCE_LENGTH,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    SENTIMENT_MODEL,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)

torch.manual_seed(RANDOM_STATE)


# ═══════════════════════════════════════════════════════════════════
# 1. LSTM CLASSIFIER
# ═══════════════════════════════════════════════════════════════════
class LSTMClassifier(nn.Module):
    """Bidirectional LSTM for sequential feature classification.

    Advantages
    ----------
    - Models temporal dependencies in price sequences.
    - Bidirectional attention captures past & future context.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)                     # (B, T, 2H)
        attn_weights = torch.softmax(
            self.attention(lstm_out), dim=1
        )                                               # (B, T, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (B, 2H)
        return self.classifier(context)                 # (B, C)


# ═══════════════════════════════════════════════════════════════════
# 2. FinBERT TRANSFORMER CLASSIFIER
# ═══════════════════════════════════════════════════════════════════
class FinBERTClassifier(nn.Module):
    """Fine-tuned FinBERT for 3-class stock-impact prediction.

    Advantages
    ----------
    - Pre-trained on financial corpus — understands domain jargon.
    - Transfer learning yields strong results with limited labels.
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(SENTIMENT_MODEL)
        hidden = self.bert.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]   # [CLS]
        return self.classifier(cls_output)


# ═══════════════════════════════════════════════════════════════════
# 3. DATASET WRAPPERS
# ═══════════════════════════════════════════════════════════════════
class TabularDataset(Dataset):
    """Simple dataset for LSTM from tabular feature arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,D)
        if seq_len > 1:
            self.X = self.X.repeat(1, seq_len, 1)  # repeat for sequence dim
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TextDataset(Dataset):
    """Tokenised text dataset for FinBERT."""

    def __init__(self, texts: list, labels: np.ndarray):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        self.encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": self.labels[idx],
        }


# ═══════════════════════════════════════════════════════════════════
# 4. TRAINING HELPERS
# ═══════════════════════════════════════════════════════════════════
def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
) -> LSTMClassifier:
    """Train the LSTM classifier and return the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(input_dim=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total if total else 0
        logger.info(
            "Epoch %d/%d — loss: %.4f — val_acc: %.4f",
            epoch + 1, epochs, total_loss / len(train_loader), val_acc,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    logger.info("LSTM training complete. Best val acc: %.4f", best_val_acc)
    return model


def train_finbert(
    texts_train: list,
    y_train: np.ndarray,
    texts_val: list,
    y_val: np.ndarray,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
) -> FinBERTClassifier:
    """Fine-tune FinBERT classifier and return trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FinBERTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    train_ds = TextDataset(texts_train, y_train)
    val_ds = TextDataset(texts_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                preds = model(input_ids, attention_mask).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total else 0
        logger.info(
            "[FinBERT] Epoch %d/%d — loss: %.4f — val_acc: %.4f",
            epoch + 1, epochs, total_loss / len(train_loader), val_acc,
        )

    return model
