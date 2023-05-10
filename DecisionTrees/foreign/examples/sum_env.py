from core import REINFORCELearner
from foreign import ForeignEnv

env = ForeignEnv(["./cpp_utils/Example_one"])
learner = REINFORCELearner(env, use_critic=True)
learner.fit()
tree = learner.generate_tree()