from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from .description import (ContinuousParameterDescription,
                          DiscreteParameterDescription, NormalizedParameters,
                          ParametersDescription)


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
                column = np.vectorize(lambda p_v: decoder.decode(round(p_v)))(values[:, column_id, :])
            elif isinstance(decoder, ContinuousParameterDescription):
                column = np.vectorize(lambda p_v: decoder.scale(p_v))(values[:, column_id, :])
            else:
                raise ValueError("This decoder is not supported")
            decoded_columns[p_name] = column.T.tolist()

        with open(filename, "w") as file:
            file.write(f"Node count\n{tree.node_count}")

            def write_array(name, arr, sep=" "):
                file.write(f"\n{name}\n")
                as_str = map(str, arr)
                file.write(sep.join(as_str))

            write_array("Children left", tree.children_left)
            write_array("Children right", tree.children_right)
            write_array("Feature", tree.feature)
            write_array("Threshold", tree.threshold)

            file.write("\nValues")
            for f_name, f_list in decoded_columns.items():
                file.write(f"\n{f_name}\n")
                file.write("\n".join(map(str, f_list[0])))

        return super().save(filename)
