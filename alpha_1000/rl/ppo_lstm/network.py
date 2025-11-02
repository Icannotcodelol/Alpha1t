"""Neural network architecture for PPO-LSTM agent."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from ..encoding import CARDS_DIM, MAX_HAND_CARDS

__all__ = ["TysiacNetwork"]


class TysiacNetwork(nn.Module):
    """Multi-head policy and value network with attention and LSTM."""

    def __init__(self, hidden_size: int = 256, lstm_layers: int = 2) -> None:
        super().__init__()
        self.card_encoder = nn.Sequential(nn.Linear(CARDS_DIM, hidden_size), nn.ReLU())
        self.trick_encoder = nn.Sequential(nn.Linear(CARDS_DIM, hidden_size), nn.ReLU())
        self.meta_encoder = nn.Sequential(nn.Linear(20, hidden_size), nn.ReLU())
        self.history_lstm = nn.LSTM(hidden_size, hidden_size, lstm_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=0.1, batch_first=True)
        fusion_dim = hidden_size * 3
        self.fusion = nn.Sequential(nn.Linear(fusion_dim, hidden_size), nn.ReLU(), nn.Dropout(0.1))
        self.bid_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 20))
        self.play_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, MAX_HAND_CARDS))
        self.bomb_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 2))
        self.value_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run a forward pass returning logits for each head and the value."""

        hand_summary = self._encode_hand(inputs["hand"], inputs["hand_mask"])
        trick_context = self._encode_trick(inputs["current_trick"], inputs["current_trick_mask"])
        history_context = self._encode_history(inputs)
        meta = self._encode_meta(inputs)
        sequence_context = history_context + trick_context
        combined = torch.cat([hand_summary, sequence_context, meta], dim=-1)
        features = self.fusion(combined)
        outputs = {
            "bid_logits": self.bid_head(features),
            "play_logits": self.play_head(features),
            "bomb_logits": self.bomb_head(features),
            "value": self.value_head(features).squeeze(-1),
        }
        return outputs

    def _encode_hand(self, hand: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return pooled hand embedding."""

        embedded = self.card_encoder(hand)
        weights = mask.unsqueeze(-1)
        summed = (embedded * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def _encode_trick(self, trick: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return embedding for current trick."""

        embedded = self.trick_encoder(trick)
        weights = mask.unsqueeze(-1)
        summed = (embedded * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def _encode_history(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate trick history using an LSTM with self-attention."""

        history = inputs["trick_history"]
        mask = inputs["trick_history_mask"]
        batch, tricks, cards, dim = history.shape
        encoded = self.trick_encoder(history.view(batch * tricks * cards, dim))
        encoded = encoded.view(batch, tricks, cards, -1)
        weights = mask.unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=2)
        denom = weights.sum(dim=2).clamp_min(1.0)
        pooled = pooled / denom
        lstm_out, _ = self.history_lstm(pooled)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return attn_out[:, -1]

    def _encode_meta(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse scalar meta features with meld and bomb indicators."""

        melds = inputs["melds"].reshape(inputs["melds"].shape[0], -1)
        bombs = inputs["bombs"]
        scores = inputs["scores"]
        meta = inputs["meta"]
        stacked = torch.cat([melds, bombs, scores, meta], dim=-1)
        return self.meta_encoder(stacked)
