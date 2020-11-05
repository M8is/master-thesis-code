import torch

from numpy.random import weibull
from math import pi, sqrt
from mc_estimators.probabilistic_objective_gradient import Probabilistic


class MVD(Probabilistic):
    def __init__(self, objective, episode_size, output_dim, coupled=False):
        super().__init__()
        self.objective = objective
        self.episode_size = episode_size
        self.output_dim = output_dim
        self.coupled = coupled
    
    def get_samples(self):
        pos_samples = torch.Tensor(weibull(sqrt(2.), (self.episode_size, self.output_dim)))
        if self.coupled:
            neg_samples = pos_samples
        else:
            neg_samples = torch.Tensor(weibull(sqrt(2.), (self.episode_size, self.output_dim)))
        return pos_samples, neg_samples
        
    def forward_mc(self, params):
        mean, cov = params
        pos_samples, neg_samples = self.get_samples()
        mean_vector = torch.stack([mean] * self.episode_size, dim=0)
        if self.output_dim > 1:
            l = torch.cholesky(cov)
            pos_losses = self.objective(mean_vector + pos_samples @ l)
            neg_losses = self.objective(mean_vector - neg_samples @ l)
            c = torch.inverse(sqrt(2*pi) * cov)
            grad_estimate = ((pos_losses - neg_losses) @ c).mean(dim=0)
        else:
            pos_losses = self.objective(mean_vector + pos_samples * cov)
            neg_losses = self.objective(mean_vector - neg_samples * cov)
            c = 1 / (sqrt(2 * pi) * cov)
            grad_estimate = ((pos_losses - neg_losses) * c).mean(dim=0)
        return grad_estimate.detach() * mean
