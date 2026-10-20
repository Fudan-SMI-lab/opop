"""Two-layer mean GraphSAGE, expressed using ordinary PyTorch operations."""
import torch
from torch import nn


class MeanSAGE(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.project = nn.Linear(2 * input_features, output_features)

    def forward(self, features, sources, destinations, degree):
        messages = features.index_select(0, sources)
        neighbors = torch.zeros_like(features).index_add_(0, destinations, messages)
        neighbors = neighbors / degree.clamp_min(1).unsqueeze(1)
        return self.project(torch.cat((features, neighbors), dim=1))


class GraphEncoder(nn.Module):
    def __init__(self, input_features=128, hidden_features=256, output_features=64):
        super().__init__()
        self.first = MeanSAGE(input_features, hidden_features)
        self.second = MeanSAGE(hidden_features, output_features)

    def forward(self, features, sources, destinations, degree):
        hidden = torch.relu(self.first(features, sources, destinations, degree))
        return self.second(hidden, sources, destinations, degree)
