from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
from typing import Dict, Any, List
import pandas as pd
import json
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from .description import NormalizedParameters, ParametersDescription, DiscreteParameterDescription, ContinuousParameterDescription


class IDecisionTree(ABC):
    @abstractmethod
    def predict(self, state: ndarray) -> Dict[str, Any]:
        ...

    @abstractmethod
    def save(self, filename: str):
        ...


class SklearnDecisionTree(IDecisionTree):
    def __init__(self,
                 states: List[ndarray],
                 parameters: List[NormalizedParameters],
                 param_description: ParametersDescription):
        self._param_description = param_description

        X = pd.DataFrame(states)

        param_dicts = []
        for params in parameters:
            p_d = {}
            p_d.update(params.discrete)
            p_d.update(params.continuous)
            param_dicts.append(p_d)
        y = pd.DataFrame(param_dicts)
        self._p_names = y.columns

        tree_hyperopt_space = {
            "max_depth": [3, 5, 7, 10],
            "min_samples_leaf": [1, 10, 20, 40, 70]
        }
        grid_search = GridSearchCV(DecisionTreeRegressor(), tree_hyperopt_space)
        grid_search.fit(X, y)
        best_tree_params = grid_search.best_params_

        self._tree = DecisionTreeRegressor(**best_tree_params)
        self._tree.fit(X, y)

    def predict(self, state: ndarray) -> Dict[str, Any]:
        prediction = self._tree.predict(state.reshape(1, -1))
        prediction = pd.DataFrame(prediction, columns=self._p_names).to_dict("records")[0]
        decoded_params = {}
        for name, param_decoder in self._param_description.items():
            if isinstance(param_decoder, DiscreteParameterDescription):
                decoded_params[name] = param_decoder.decode(int(prediction[name]))
            elif isinstance(param_decoder, ContinuousParameterDescription):
                decoded_params[name] = param_decoder.scale(prediction[name])
        return decoded_params

    def save(self, filename: str):
        tree = self._tree.tree_

        values = tree.value
        decoded_columns = {}
        for column_id, p_name in enumerate(self._p_names):
            decoder = self._param_description.get_description(p_name)
            if isinstance(decoder, DiscreteParameterDescription):
                column = np.vectorize(lambda p_v: decoder.decode(p_v))(values[:, column_id, :])
            elif isinstance(decoder, ContinuousParameterDescription):
                column = np.vectorize(lambda p_v: decoder.scale(p_v))(values[:, column_id, :])
            else:
                raise ValueError("This decoder is not supported")
            decoded_columns[p_name] = column.tolist()
            
        as_json = {
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "value": decoded_columns
        }

        with open(filename, 'w') as file:
            json.dump(as_json, file, indent=2)
