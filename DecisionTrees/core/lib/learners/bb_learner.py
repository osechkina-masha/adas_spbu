from abc import abstractmethod

from tqdm import trange

from .interface import Learner
from ..decision_tree import IDecisionTree, SklearnDecisionTree
from ..description import NormalizedParameters
from ..environment import Environment


class BlackBoxOptimizerLearner(Learner):
    def __init__(self, env: Environment) -> None:
        self._env = env

    @abstractmethod
    def optimize(self, env: Environment) -> NormalizedParameters:
        ...

    def fit(self, iterations: int = 10_000):
        states = []
        params = []
        for i in trange(iterations, desc=f"Running {type(self).__name__}"):
            self._env.reset()
            p = self.optimize(self._env)
            states.append(self._env.current_state())
            params.append(p)
        self._tree = SklearnDecisionTree(states, params, self._env.parameters_description)

    def generate_tree(self) -> IDecisionTree:
        return self._tree
