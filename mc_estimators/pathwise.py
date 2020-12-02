import torch
from torch.distributions import MultivariateNormal

from .base import MultivariateNormalProbabilistic


class MultivariateNormalPathwise(MultivariateNormalProbabilistic):
    def grad_samples(self, params):
        mean, log_cov = params
        cov = torch.diag(torch.exp(log_cov.squeeze()))
        return MultivariateNormal(mean, covariance_matrix=cov).rsample((self.sample_size,))

    def backward(self, losses):
        return losses.mean().backward()