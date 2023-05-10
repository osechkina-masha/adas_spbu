from typing import Any, Dict

import torch
from numpy import ndarray

from ...description import NormalizedParameters, ParametersDescription
from .model import Policy, REINFORCEModel


def gready_sampling(policy: Policy) -> NormalizedParameters:
    discrete = {}
    for p_name, p_vector in policy.discrete.items():
        discrete[p_name] = torch.argmax(p_vector).item()
    continuous = {}
    for p_name, p_mean in policy.mean.items():
        continuous[p_name] = p_mean.item()
    return NormalizedParameters(discrete, continuous)


def inference_reinforce(model: REINFORCEModel, state: ndarray, param_desc: ParametersDescription) -> Dict[str, Any]:
    policy = model(torch.from_numpy(state).float())
    action = gready_sampling(policy)
    parameters = param_desc.decode_parameters(action)
    return parameters
