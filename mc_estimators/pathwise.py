import torch
from torch.distributions import MultivariateNormal

from .base import MultivariateNormalProbabilistic


class MultivariateNormalPathwise(MultivariateNormalProbabilistic):
    def grad_samples(self, params):
        mean, log_std = params
        cov = torch.diag_embed(torch.exp(2 * log_std))
        return MultivariateNormal(mean, cov).rsample((self.sample_size,))

    def backward(self, losses):
        losses.mean().backward()
