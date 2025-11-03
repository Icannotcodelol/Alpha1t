"""PPO-LSTM agent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..encoding import Observation
from .network import TysiacNetwork

__all__ = ["AgentOutput", "PpoLstmAgent"]


@dataclass(frozen=True)
class AgentOutput:
    """Container for sampled actions and diagnostics."""

    actions: Dict[str, torch.Tensor]
    log_probs: Dict[str, torch.Tensor]
    value: torch.Tensor

    def __repr__(self) -> str:  # pragma: no cover - representation utility
        return f"AgentOutput(actions={list(self.actions)}, value={float(self.value.mean())})"


@dataclass
class PpoLstmAgent:
    """PPO agent managing inference and optimisation."""

    network: TysiacNetwork
    optimizer: torch.optim.Optimizer
    device: torch.device
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None

    def __repr__(self) -> str:  # pragma: no cover - trivial helper
        return f"PpoLstmAgent(network={self.network.__class__.__name__})"

    @classmethod
    def create(cls, learning_rate: float = 3e-4, device: torch.device | None = None) -> "PpoLstmAgent":
        """Factory creating a default agent with Adam optimiser."""

        net = TysiacNetwork()
        dev = device or torch.device("cpu")
        net.to(dev)
        # Use AdamW with weight decay for regularization
        optimiser = torch.optim.AdamW(
            net.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4
        )
        # Add cosine annealing scheduler for gradual LR decay
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=10000,  # Reset every 10k iterations
            eta_min=learning_rate * 0.1
        )
        return cls(network=net, optimizer=optimiser, device=dev, scheduler=scheduler)

    def act(
        self,
        observation: Observation,
        action_masks: Mapping[str, np.ndarray] | None = None,
        *,
        greedy: bool = False,
    ) -> AgentOutput:
        """Sample actions from the policy applying optional masks."""

        torch_inputs = self._to_torch(observation)
        outputs = self.network(torch_inputs)
        head_masks = self._prepare_masks(action_masks, torch_inputs["hand"].shape[0])
        actions: Dict[str, torch.Tensor] = {}
        log_probs: Dict[str, torch.Tensor] = {}
        for head_name in ("bid", "play", "bomb"):
            logits = outputs[f"{head_name}_logits"]
            masked_logits = self._apply_mask(logits, head_masks.get(head_name))
            dist = Categorical(logits=masked_logits)
            action = torch.argmax(masked_logits, dim=-1) if greedy else dist.sample()
            actions[head_name] = action
            log_probs[head_name] = dist.log_prob(action)
        value = outputs["value"]
        return AgentOutput(actions=actions, log_probs=log_probs, value=value)

    def update(self, loss: torch.Tensor) -> None:
        """Backpropagate a PPO loss and apply gradient clipping."""

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients by value AND norm for double protection
        torch.nn.utils.clip_grad_value_(self.network.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def _to_torch(self, observation: Observation) -> Dict[str, torch.Tensor]:
        """Convert numpy observation tensors into torch tensors on device."""

        torch_inputs: Dict[str, torch.Tensor] = {}
        for key, value in observation.tensors.items():
            tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            torch_inputs[key] = tensor.unsqueeze(0)
        return torch_inputs

    def _prepare_masks(
        self, masks: Mapping[str, np.ndarray] | None, batch: int
    ) -> Dict[str, torch.Tensor]:
        """Convert optional masks to tensors."""

        prepared: Dict[str, torch.Tensor] = {}
        if not masks:
            return prepared
        for key, value in masks.items():
            tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[0] != batch:
                tensor = tensor.expand(batch, *tensor.shape[1:])
            prepared[key] = tensor
        return prepared

    def _apply_mask(self, logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Apply additive mask, setting invalid logits to -inf."""

        if mask is None:
            return logits
        invalid = (mask <= 0).to(dtype=logits.dtype)
        penalty = (1.0 - mask) * 1e10
        return logits - invalid * penalty
