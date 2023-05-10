from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...description import (ContinuousParameterDescription,
                            DiscreteParameterDescription,
                            ParametersDescription)


@dataclass
class Policy:
    discrete: Dict[str, Tensor]
    mean: Dict[str, Tensor]
    std: Dict[str, Tensor]


class REINFORCEModel(nn.Module):
    def __init__(self,
                 parameters: ParametersDescription,
                 inp_dim: int,
                 hidden_dim: int = 512,
                 n_common_layers: int = 2,
                 with_critic: bool = False,
                 critic_hidden_dim: int = 512,
                 n_critic_hidden_layers: int = 1) -> None:
        super().__init__()

        self._param_names = [p[0] for p in parameters.items()]

        if n_common_layers == 0:
            hidden_dim = inp_dim
            self.common = nn.Sequential()
        else:
            layers = [nn.Linear(inp_dim, hidden_dim), nn.ReLU()]
            for i in range(1, n_common_layers):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                ])
            self.common = nn.Sequential(*layers)

        self.discrete_layers = nn.ModuleDict()
        self.continuous_layers_means = nn.ModuleDict()
        self.continuous_layers_std = nn.ModuleDict()

        for p_name, p_desc in parameters.items():
            if isinstance(p_desc, DiscreteParameterDescription):
                self.discrete_layers[p_name] = nn.Linear(hidden_dim, p_desc.n_categories)
            elif isinstance(p_desc, ContinuousParameterDescription):
                self.continuous_layers_means[p_name] = nn.Linear(hidden_dim, 1)
                self.continuous_layers_std[p_name] = nn.Linear(hidden_dim, 1)

        if with_critic:
            critic_layers = [nn.Linear(inp_dim + hidden_dim, critic_hidden_dim), nn.ReLU()]
            for _ in range(n_critic_hidden_layers):
                critic_layers.extend([
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
                ])
            critic_layers.append(nn.Linear(critic_hidden_dim, 1))
            self.critic = nn.Sequential(*critic_layers)

    def forward(self, x: Tensor) -> Policy:
        x = self.common(x)

        discrete_policy = {}
        for p_name, p_layer in self.discrete_layers.items():
            discrete_policy[p_name] = F.softmax(p_layer(x), dim=-1)

        continuous_mean_policy = {}
        continuous_std_policy = {}
        for p_name in self.continuous_layers_means.keys():
            mean_layer = self.continuous_layers_means[p_name]
            std_layer = self.continuous_layers_std[p_name]

            mean_policy = torch.sigmoid(mean_layer(x))
            std_policy = F.softplus(std_layer(x))

            continuous_mean_policy[p_name] = mean_policy
            continuous_std_policy[p_name] = std_policy

        return Policy(discrete_policy,
                      continuous_mean_policy,
                      continuous_std_policy)

    def __call__(self, x: Tensor) -> Policy:
        return super().__call__(x)
