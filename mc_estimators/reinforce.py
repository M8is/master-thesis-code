import torch
from torch.distributions import MultivariateNormal

from .base import MultivariateNormalProbabilistic


class MultivariateNormalReinforce(MultivariateNormalProbabilistic):
    def grad_samples(self, params):
        mean, log_std = params
        cov = torch.diag_embed(torch.exp(2 * log_std))
        samples = MultivariateNormal(mean, cov).sample((self.sample_size,))
        self._to_backward((mean, cov, samples))
        return samples

    def backward(self, losses):
        mean, cov, samples = self._from_forward()
        log_probs = MultivariateNormal(mean, cov).log_prob(samples)
        (losses * -log_probs).mean().backward()
