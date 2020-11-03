import torch

from probabilistic_objective_gradient import ProbabilisticObjective


class ReinforceNormal(ProbabilisticObjective):
    def __init__(self, objective, episode_size, init_mean, init_cov):
        super().__init__(init_mean, init_cov)
        self.objective = objective
        self.episode_size = episode_size
        self.distribution = torch.distributions.MultivariateNormal
        self.losses = []
    
    def get_samples(self, x):
        dist = self.distribution(self.mean, self.cov)
        return dist.sample((self.episode_size,))
        
    def gradient_estimate(self, x):
        samples = self.get_samples(x)
        loss = self.objective(samples)
        log_derivative = torch.inverse(self.cov) * (samples - self.mean)
        mean_surrogate_loss = (loss * log_derivative).mean(dim=0)
        self.losses.append(mean_surrogate_loss)
        return mean_surrogate_loss
