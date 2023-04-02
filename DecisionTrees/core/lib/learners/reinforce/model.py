import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ...description import ParametersDescription, DiscreteParameterDescription, ContinuousParameterDescription
from typing import Dict, Optional
from dataclasses import dataclass
import random


@dataclass
class Policy:
    discrete: Dict[str, Tensor]
    mean: Dict[str, Tensor]
    std: Dict[str, Tensor]


class REINFORCEModel(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim: int, parameters: ParametersDescription) -> None:
        super().__init__()

        self._param_names = [p[0] for p in parameters.items()]
        self.common_layer_l1 = nn.Linear(inp_dim, hidden_dim)
        self.common_layer_l2 = nn.Linear(hidden_dim, hidden_dim)

        self.discrete_layers = nn.ModuleDict()
        self.continuous_layers_means = nn.ModuleDict()
        self.continuous_layers_std = nn.ModuleDict()

        for p_name, p_desc in parameters.items():
            if isinstance(p_desc, DiscreteParameterDescription):
                self.discrete_layers[p_name] = nn.Linear(hidden_dim, p_desc.n_categories)
            elif isinstance(p_desc, ContinuousParameterDescription):
                self.continuous_layers_means[p_name] = nn.Linear(hidden_dim, 1)
                self.continuous_layers_std[p_name] = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor, grad_parameter: Optional[str] = None) -> Policy:
        if grad_parameter is None:
            grad_parameter = random.choice(self._param_names)

        x = F.relu(self.common_layer_l1(x))
        x = F.relu(self.common_layer_l2(x))

        discrete_policy = {}
        for p_name, p_layer in self.discrete_layers.items():
            y = x
            if p_name != grad_parameter:
                y = x.detach()

            discrete_policy[p_name] = F.softmax(p_layer(y), dim=-1)

        continuous_mean_policy = {}
        continuous_std_policy = {}
        for p_name in self.continuous_layers_means.keys():
            y = x
            if p_name != grad_parameter:
                y = x.detach()

            mean_layer = self.continuous_layers_means[p_name]
            std_layer = self.continuous_layers_std[p_name]

            mean_policy = torch.sigmoid(mean_layer(y))
            std_policy = F.softplus(std_layer(y))

            continuous_mean_policy[p_name] = mean_policy
            continuous_std_policy[p_name] = std_policy

        return Policy(discrete_policy, continuous_mean_policy, continuous_std_policy)

    def __call__(self, x: Tensor) -> Policy:
        return super().__call__(x)
