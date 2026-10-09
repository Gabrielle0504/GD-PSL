"""
A simple FC Pareto Set model.
"""

import torch
import torch.nn as nn

from experiment_config import DEFAULT_CONFIG


DEFAULT_HIDDEN_WIDTH = DEFAULT_CONFIG.hidden_width


class ParetoSetModel(torch.nn.Module):
    """MLP baseline mapping objective preferences to normalized decisions."""

    def __init__(self, n_dim: int, n_obj: int, hidden_width: int = DEFAULT_HIDDEN_WIDTH):
        super(ParetoSetModel, self).__init__()
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        self.n_dim = n_dim
        self.n_obj = n_obj
        # 1024 is the original wide-baseline capacity; expose it as a parameter
        # so experiments can tune model size without editing the architecture.
        self.n_node = hidden_width

        self.fc1 = nn.Linear(self.n_obj, self.n_node)
        self.fc2 = nn.Linear(self.n_node, self.n_node)
        self.fc3 = nn.Linear(self.n_node, self.n_dim)

    def forward(self, pref):

        x = torch.relu(self.fc1(pref))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        x = torch.sigmoid(x)

        return x.to(torch.float64)
