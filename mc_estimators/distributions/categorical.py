import torch
from .distribution_base import Distribution


class Categorical(Distribution):
    def sample(self, params, size=1, with_grad=False):
        return torch.distributions.Categorical(params).sample((size,))

    def mvd_sample(self, params, size):
        return torch.diag_embed(torch.tensor(range(params.size(-1))))

    def mvd_constant(self, params):
        return 1 / params.size(-1)

    def kl(self, params):
        p = torch.distributions.Categorical(params)
        q = torch.distributions.Categorical(torch.ones_like(params).mean(-1, keepdim=True))
        return torch.distributions.kl_divergence(p, q)

    def log_prob(self, params, samples):
        return torch.distributions.Categorical(params).log_prob(samples)
