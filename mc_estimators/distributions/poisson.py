import torch
from .exponential import Exponential


class Poisson(Exponential):
    def sample(self, raw_params, size=1, with_grad=False):
        if with_grad:
            raise ValueError("Poisson cannot be reparameterized.")
        else:
            params = self._as_rate(raw_params)
            dist = torch.distributions.poisson.Poisson(params)
            return dist.sample((size,)).to(self.device)

    def mvd_sample(self, raw_params, size):
        rate = self._as_rate(raw_params)
        with torch.no_grad():
            pos_samples = self.__sample_poisson(size, rate + 1.)
            neg_samples = self.__sample_poisson(size, rate)
            samples = torch.diag_embed(torch.stack((pos_samples, neg_samples))).transpose(2, 3)
            return samples + rate

    def log_prob(self, raw_params, samples):
        rate = self._as_rate(raw_params)
        return torch.distributions.Poisson(rate).log_prob(samples).sum(dim=-1).to(self.device)

    def __sample_poisson(self, sample_size, rate):
        return torch.distributions.poisson.Poisson(rate).sample((sample_size,)).to(self.device)
