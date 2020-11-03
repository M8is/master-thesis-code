import torch

from numpy.random import weibull
from math import pi, sqrt
from grad_utils import doublesided_maxwell
from probabilistic_objective_gradient import ProbabilisticObjective


class MVDNormal(ProbabilisticObjective):
    def __init__(self, objective, episode_size, init_mean, init_cov, coupled=False):
        super().__init__(init_mean, init_cov)
        self.objective = objective
        self.episode_size = episode_size
        self.coupled = coupled
        self.losses = []
    
    def get_samples(self, x):
        mean_size = self.mean.size()[0]
        pos_samples = torch.Tensor(weibull(sqrt(2.), (self.episode_size, mean_size)))
        if self.coupled:
            neg_samples = pos_samples
        else:
            neg_samples = torch.Tensor(weibull(sqrt(2.), (self.episode_size, mean_size)))
        return pos_samples, neg_samples
        
    def gradient_estimate(self, x):
        pos_samples, neg_samples = self.get_samples(x)
        l = torch.cholesky(self.cov)
        scaled_pos = torch.matmul(pos_samples, l)
        scaled_neg = torch.matmul(neg_samples, l)
        pos_losses = self.objective(torch.stack([self.mean] * pos_samples.size()[0], dim=0) + scaled_pos)
        neg_losses = self.objective(torch.stack([self.mean] * neg_samples.size()[0], dim=0) - scaled_neg)
        mean_diff = (pos_losses - neg_losses).mean(dim=0)
        c = torch.inverse(sqrt(2*pi) * self.cov)
        surrogate_loss = c * mean_diff
        self.losses.append(surrogate_loss)        
        return surrogate_loss
