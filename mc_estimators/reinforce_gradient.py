from mc_estimators.probabilistic_objective_gradient import Probabilistic


class Reinforce(Probabilistic):
    def __init__(self, objective, episode_size, distribution):
        super().__init__()
        self.objective = objective
        self.episode_size = episode_size
        self.distribution = distribution
        
    def forward_mc(self, params):
        dist = self.distribution(*params)
        samples = dist.sample((self.episode_size,))
        losses = self.objective(samples)
        log_probs = dist.log_prob(samples).reshape(-1, 1)
        return (losses * log_probs).mean(dim=0)
