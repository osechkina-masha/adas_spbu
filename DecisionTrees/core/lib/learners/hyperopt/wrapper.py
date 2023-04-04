from hyperopt import hp, fmin, tpe
from ..bb_learner import BlackBoxOptimizerLearner
from ...environment import Environment
from ...description import NormalizedParameters, ContinuousParameterDescription, DiscreteParameterDescription


class HyperOptLearner(BlackBoxOptimizerLearner):
    def __init__(self, env: Environment, max_evals: int=100) -> None:
        super().__init__(env)
        self._max_evals = max_evals

        self._space = {"discrete": {}, "continuous": {}}
        for p_name, p_desc in env.parameters_description.items():
            if isinstance(p_desc, ContinuousParameterDescription):
                self._space["continuous"][p_name] = hp.uniform(p_name, 0, 1)
            elif isinstance(p_desc, DiscreteParameterDescription):
                self._space["discrete"][p_name] = hp.choice(p_name, list(range(p_desc.n_categories)))

        def target_func(params):
            params = NormalizedParameters(params["discrete"], params["continuous"])
            params = env.parameters_description.decode_parameters(params)
            return -1 * env.score(params)
        self._target_func = target_func

    def optimize(self, env: Environment) -> NormalizedParameters:
        optim_params = fmin(
            self._target_func,
            algo=tpe.suggest,
            space=self._space,
            max_evals=self._max_evals,
            verbose=0,
            return_argmin=False
        )

        return NormalizedParameters(optim_params["discrete"], optim_params["continuous"])
