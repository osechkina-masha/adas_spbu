from ..interface import Learner
from ...decision_tree import SklearnDecisionTree, IDecisionTree
from ...environment import Environment
from .train import run_genetic
from tqdm import trange


class GeneticLearner(Learner):
    def __init__(self,
                 env: Environment,
                 n_generations: int = 7, 
                 pop_size: int = 20,
                 tourn_size: int = 2,
                 p_crossover: float = 0.9,
                 p_mutation: float = 0.1
                 ) -> None:
        self._env = env
        self._n_generations = n_generations
        self._pop_size = pop_size
        self._tourn_size = tourn_size
        self._p_crossover = p_crossover
        self._p_mutation = p_mutation

    def fit(self, iterations: int = 10_000):
        states = []
        params = []
        for i in trange(iterations, desc="Running Genetic learner"):
            self._env.reset()
            p = run_genetic(self._env,
                            self._n_generations,
                            self._pop_size,
                            self._tourn_size,
                            self._p_crossover,
                            self._p_mutation)
            
            states.append(self._env.current_state())
            params.append(p)
        self._tree = SklearnDecisionTree(states, params, self._env.parameters_description)

    def generate_tree(self) -> IDecisionTree:
        return self._tree