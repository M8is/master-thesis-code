import torch

from mc_estimators.probabilistic_objective_gradient import Probabilistic


class Pathwise(Probabilistic):
    def __init__(self, objective, episode_size, distribution):
        super().__init__()
        self.objective = objective
        self.episode_size = episode_size
        self.distribution = distribution

    def forward_mc(self, params):
        samples = self.distribution(*params).rsample((self.episode_size,))
        return self.objective(samples).mean(dim=0)
