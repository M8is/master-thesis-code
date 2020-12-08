import torch
from torch.distributions import MultivariateNormal

from .base import MultivariateNormalProbabilistic


class MultivariateNormalPathwise(MultivariateNormalProbabilistic):
    def grad_samples(self, params):
        mean, std = params
        cov = torch.diagflat(std ** 2 + 1e-10)
        return MultivariateNormal(mean, cov).rsample((self.sample_size,)).squeeze(1)

    def backward(self, losses):
        return losses.mean().backward()
