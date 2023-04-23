import os
from typing import Optional

import torch
from torch import Tensor
from torch.optim import Adam
from tqdm import trange
from torch import nn

from ...description import NormalizedParameters
from ...environment import Environment
from .model import Policy, REINFORCEModel


def sample_from_policy(policy: Policy) -> NormalizedParameters:
    discrete_params = {}
    for p_name, p_vector in policy.discrete.items():
        discrete_params[p_name] = torch.distributions.Categorical(probs=p_vector).sample().item()
    continuous_params = {}
    for p_name, p_mean in policy.mean.items():
        p_std = policy.std[p_name]
        continuous_params[p_name] = torch.distributions.Normal(p_mean, p_std).sample().clip(0, 1).item()
    return NormalizedParameters(discrete_params, continuous_params)


def discrete_logprob(dist: Tensor, action: int) -> Tensor:
    return torch.log(dist[action])


# this is not an actual probability, but it will suit well during training
def normal_logprob(mu_v: Tensor, var_v: Tensor, actions_v: Tensor) -> Tensor:
    p1 = -((mu_v - actions_v) ** 2 / (2 * var_v.clamp(min=1e-3)))
    p2 = -torch.log(2 * var_v * torch.pi) / 2
    return p1 + p2


def discrete_entropy(dist: Tensor) -> Tensor:
    return -torch.sum(dist * torch.log(dist))


def normal_entropy(var_v: Tensor) -> Tensor:
    return torch.log(torch.e * torch.pi * var_v) / 2


def train_reinforce(model: REINFORCEModel,
                    env: Environment,
                    epochs: int,
                    batch_size: int = 64,
                    save_model_every_n_steps: int = 1000,
                    checkpoint_path: Optional[str] = None,
                    discrete_entropy_beta: float = 0.05,
                    continuous_entropy_beta: float = 0.05):
    optimizer = Adam(params=model.parameters())
    optimizer.zero_grad()

    # # Gradient clipping
    # for p in model.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

    critic = None
    critic_optimizer = None
    with_critic = hasattr(model, "critic")
    if with_critic:
        critic = model.critic
        critic_optimizer = Adam(params=critic.parameters())

    # loss_sum = torch.Tensor([0])
    for episode in trange(epochs, desc="Training REINFORCE"):
        # Slowly decreasing entropy multiplier
        entropy_mult = ((epochs - episode) / epochs)

        # Forward pass
        env.reset()
        state = torch.from_numpy(env.current_state()).float()
        policy = model(state)
        action = sample_from_policy(policy)
        parameters = env.parameters_description.decode_parameters(action)
        reward = env.score(parameters)

        if with_critic:
            original_reward = reward
            x = torch.cat((state, model.common(state).detach()))
            critic_prediction = critic(x)
            critic_loss = nn.functional.huber_loss(critic_prediction, torch.Tensor([original_reward]))
            critic_loss.backward()
            reward -= critic_prediction.item()

        # Calculating loss
        loss = torch.Tensor([0]).float()
        for name, sub_action in policy.discrete.items():
            loss += - discrete_logprob(sub_action, action.discrete[name]) * reward
            loss += - entropy_mult * discrete_entropy_beta * discrete_entropy(sub_action)
        for name, mu_v in policy.mean.items():
            std_v = policy.std[name]
            var_v = std_v ** 2
            var_v = var_v.clamp(min=1e-3)
            loss += - normal_logprob(mu_v, var_v, torch.Tensor([action.continuous[name]])) * reward
            loss += - entropy_mult * continuous_entropy_beta * normal_entropy(var_v)
        loss /= batch_size
        loss.backward()

        # Backward pass
        if episode % batch_size == 0 and episode != 0:
            optimizer.step()
            optimizer.zero_grad()
            if with_critic:
                critic_optimizer.step()
                critic_optimizer.zero_grad()

            # Saving model if checkpoint path provided
            if checkpoint_path is not None and episode % (batch_size * save_model_every_n_steps) == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_path, f"epoch_{episode}.pt"))

    # Last backward pass
    optimizer.step()
