
from ...description import NormalizedParameters
from ...environment import Environment
from ..bb_learner import BlackBoxOptimizerLearner
from .train import run_genetic


class GeneticLearner(BlackBoxOptimizerLearner):
    def __init__(self,
                 env: Environment,
                 n_generations: int = 7, 
                 pop_size: int = 20,
                 tourn_size: int = 2,
                 p_crossover: float = 0.9,
                 p_mutation: float = 0.1) -> None:
        self._env = env
        self._n_generations = n_generations
        self._pop_size = pop_size
        self._tourn_size = tourn_size
        self._p_crossover = p_crossover
        self._p_mutation = p_mutation
        super().__init__(env)

    def optimize(self, env: Environment) -> NormalizedParameters:
        return run_genetic(
            env,
            self._n_generations,
            self._pop_size,
            self._tourn_size,
            self._p_crossover,
            self._p_mutation
        )