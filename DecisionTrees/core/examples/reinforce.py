from core import REINFORCELearner, Environment, ParametersDescription
import numpy as np
import random


def one_hot(value, length):
    v = np.zeros(length)
    v[value] = 1
    return v


def predict_with_tree(tree, a, b):
    vec = np.hstack([one_hot(a, 4), one_hot(b, 4)])
    prediction = tree.predict(vec)
    prediction = round(prediction['sum'])
    print(f"{a} + {b} = {prediction}")


class SimpleEnvironment(Environment):
    parameters_description = \
        ParametersDescription().add_continuous('sum', 0, 6)

    def reset(self):
        self.a = random.randint(0, 3)
        self.b = random.randint(0, 3)

    def current_state(self):
        a_vec = one_hot(self.a, 4)
        b_vec = one_hot(self.b, 4)
        return np.hstack([a_vec, b_vec])

    def score(self, parameters) -> float:
        sum = parameters['sum']
        return (6 - abs(sum - self.a - self.b)) / 6


learner = REINFORCELearner(env=SimpleEnvironment(), use_critic=True, n_epochs=10_000)
learner.fit()
tree = learner.generate_tree()

predict_with_tree(tree, 0, 0)
predict_with_tree(tree, 1, 2)
predict_with_tree(tree, 2, 3)
predict_with_tree(tree, 2, 1)
predict_with_tree(tree, 3, 3)
