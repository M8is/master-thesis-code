import torch
from .exponential import Exponential


class Poisson(Exponential):
    def sample(self, params, size=1, with_grad=False):
        if with_grad:
            raise ValueError("Poisson cannot be reparameterized.")
        else:
            dist = torch.distributions.poisson.Poisson(self.__as_rate(params))
            return dist.sample((size,)).to(self.device)

    def mvd_sample(self, params, size):
        rate = self.__as_rate(params)
        pos_samples = self.__sample_poisson(size, rate + 1.)
        neg_samples = self.__sample_poisson(size, rate)
        samples = torch.diag_embed(torch.stack((pos_samples, neg_samples))).transpose(2, 3)
        return samples + rate

    def log_prob(self, params, samples):
        return torch.distributions.Poisson(self.__as_rate(params)).log_prob(samples).sum(dim=-1).to(self.device)

    def __sample_poisson(self, sample_size, rate):
        return torch.distributions.poisson.Poisson(rate).sample((sample_size,)).to(self.device)

    @staticmethod
    def __as_rate(params):
        log_rate, = params
        rate = torch.exp(log_rate)
        return rate

