import torch
from torch.distributions import MultivariateNormal

from .base import MultivariateNormalProbabilistic


class MultivariateNormalReinforce(MultivariateNormalProbabilistic):
    def grad_samples(self, params):
        mean, std = params
        cov = torch.diagflat(std ** 2)
        samples = MultivariateNormal(mean, cov).sample((self.sample_size,)).squeeze(1)
        self._to_backward((mean, cov, samples))
        return samples

    def backward(self, losses):
        mean, cov, samples = self._from_forward()
        log_probs = MultivariateNormal(mean, cov).log_prob(samples)
        return (losses * log_probs).mean().backward()
