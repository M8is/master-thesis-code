import torch
from torch.distributions import MultivariateNormal

from .base import MultivariateNormalProbabilistic


class MultivariateNormalReinforce(MultivariateNormalProbabilistic):
    def grad_samples(self, params):
        mean, log_cov = params
        cov = torch.diag(torch.exp(log_cov.squeeze()))
        samples = MultivariateNormal(mean, cov).sample((self.sample_size,))
        self._to_backward((mean, cov, samples))
        return samples

    def backward(self, losses):
        mean, cov, samples = self._from_forward()
        log_probs = MultivariateNormal(mean, cov).log_prob(samples).reshape(-1, 1)
        return (losses * log_probs).mean().backward()
