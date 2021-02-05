import torch
from .exponential import Exponential


class Poisson(Exponential):
    def sample(self, params, size=1, with_grad=False):
        rate = self.__as_rate(params)
        if with_grad:
            return torch.poisson(rate.expand((size, *rate.shape)))
        else:
            dist = torch.distributions.poisson.Poisson(rate)
            return dist.sample((size,))

    def mvd_sample(self, params, size):
        return self.__rate_samples(size, self.__as_rate(params))

    def log_prob(self, params, samples):
        return torch.distributions.Poisson(self.__as_rate(params)).log_prob(samples).sum(dim=-1)

    @staticmethod
    def __as_rate(params):
        log_rate, = params
        rate = torch.exp(log_rate)
        return rate

    @staticmethod
    def __rate_samples(sample_size, rate):
        pos_samples = Poisson.__sample_poisson(sample_size, rate + 1.)
        neg_samples = Poisson.__sample_poisson(sample_size, rate)
        samples = torch.diag_embed(torch.stack((pos_samples, neg_samples))).transpose(2, 3)
        return samples + rate

    @staticmethod
    def __sample_poisson(sample_size, rate):
        return torch.distributions.poisson.Poisson(rate).sample((sample_size,))
