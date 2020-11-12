from mc_estimators.probabilistic_objective_gradient import Probabilistic


class Pathwise(Probabilistic):
    def __init__(self, episode_size, distribution):
        super().__init__()
        self.episode_size = episode_size
        self.distribution = distribution

    def forward_mc(self, x, objective, params):
        samples = self.distribution(*params).rsample((self.episode_size,))
        return objective(samples)
