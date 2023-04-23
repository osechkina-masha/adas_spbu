from typing import Optional

import torch

from ...decision_tree import IDecisionTree, SklearnDecisionTree
from ...environment import Environment
from ..interface import Learner
from .inference import gready_sampling
from .model import REINFORCEModel
from .train import train_reinforce


class REINFORCELearner(Learner):
    def __init__(self,
                 env: Environment,
                 n_epochs: int = 50_000,
                 hidden_dim: int = 512,
                 batch_size: int = 32,
                 save_model_every_n_steps: int = 10,
                 checkpoint_path: Optional[str] = None,
                 discrete_entropy_beta: float = 0.05,
                 continuous_entropy_beta: float = 0.05,
                 use_critic: bool = False,
                 n_model_common_layers: int = 2,
                 n_critic_layers: int = 2):
        self._env = env
        self._n_epochs = n_epochs
        self._hidden_dim = hidden_dim
        self._batch_size = batch_size
        self._save_model_every_n_steps = save_model_every_n_steps
        self._checkpoint_path = checkpoint_path
        self._discrete_entropy_beta = discrete_entropy_beta
        self._continuous_entropy_beta = continuous_entropy_beta

        # probing environment for inp_size
        env.reset()
        v = env.current_state()
        inp_size = v.shape[0]

        self._model = REINFORCEModel(
            parameters=env.parameters_description,
            inp_dim=inp_size,
            hidden_dim=hidden_dim,
            n_common_layers=n_model_common_layers,
            with_critic=use_critic,
            n_critic_hidden_layers=n_critic_layers
        )
        print("Model initialized")
        print(self._model)

        self._is_trained = False

    def fit(self):
        if self._is_trained:
            raise ValueError("Model was already trained")

        train_reinforce(
            self._model,
            self._env,
            self._n_epochs,
            self._batch_size,
            self._save_model_every_n_steps,
            self._checkpoint_path,
        )

        self._is_trained = True

    def generate_tree(self, iterations: int = 10_000) -> IDecisionTree:
        if not self._is_trained:
            raise ValueError("Tried to call generate_tree on untrained learner")
        states = []
        params = []
        with torch.no_grad():
            for i in range(iterations):
                self._env.reset()
                state = torch.from_numpy(self._env.current_state()).float()
                policy = self._model(state)
                action = gready_sampling(policy)

                states.append(state)
                params.append(action)
        tree = SklearnDecisionTree(states, params, self._env.parameters_description)
        return tree
 
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self._model.load_state_dict(checkpoint)
        self._is_trained = True