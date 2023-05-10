import numpy as np
from core import Environment, ParametersDescription

from typing import Any, Dict, List
from numpy import ndarray
from subprocess import Popen, PIPE


class ForeignEnv(Environment):
    parameters_description = ParametersDescription()

    def __init__(self, cmd_args: List[str]):
        self.process = Popen(cmd_args,
                             stdin=PIPE, stdout=PIPE, stderr=PIPE,
                             text=True, bufsize=0)

        s = self._pread_str()
        assert s == "N_discrete", s
        n_discrete = self._pread_int()

        for _ in range(n_discrete):
            p_name = self._pread_str()
            n_values = self._pread_int()
            values = [self._pread_str() for _ in range(n_values)]
            self.parameters_description.add_discrete(p_name, values)

        assert self._pread_str() == "N_continuous"
        n_continuous = self._pread_int()

        for _ in range(n_continuous):
            p_name = self._pread_str()
            min_v = self._pread_int()
            max_v = self._pread_int()
            self.parameters_description.add_continuous(p_name, min_v, max_v)

    def current_state(self) -> ndarray:
        self._pwrite("C")
        v = []
        el = self._pread_str()
        while el != "end":
            v.append(float(el))
            el = self._pread_str()
        return np.array(v)

    def score(self, parameters: Dict[str, Any]) -> float:
        self._pwrite("S")
        for p_name, p_value in parameters.items():
            self._pwrite(p_name)
            self._pwrite(p_value)
        return self._pread_float()

    def reset(self):
        self._pwrite("R")

    def _pread_str(self):
        return self.process.stdout.readline().strip()

    def _pread_int(self):
        return int(self._pread_str())

    def _pread_float(self):
        return float(self._pread_str())

    def _pwrite(self, line):
        self.process.stdin.write(f"{line}\n")
