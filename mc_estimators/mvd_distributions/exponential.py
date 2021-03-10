import torch
from .distribution_base import Distribution


class Exponential(Distribution):
    def sample(self, params, size=1, with_grad=False):
        dist = torch.distributions.exponential.Exponential(self.__as_rate(params))
        samples = dist.rsample((size,)) if with_grad else dist.sample((size,))
        return samples.to(self.device)

    def mvd_sample(self, params, size):
        rate = self.__as_rate(params)
        pos_samples = self.__sample_exponential(size, rate)
        neg_samples = self.__sample_negative(size, rate)
        samples = torch.diag_embed(torch.stack((pos_samples, neg_samples))).transpose(2, 3)
        return samples + rate

    def mvd_constant(self, params):
        return 1. / self.__as_rate(params)

    def kl(self, params):
        log_rate, = params
        return (torch.exp(log_rate) - log_rate - 1).sum(dim=1)

    def log_prob(self, params, samples):
        return torch.distributions.Exponential(self.__as_rate(params)).log_prob(samples).sum(dim=-1)

    def __sample_exponential(self, sample_size, rate):
        return torch.distributions.exponential.Exponential(rate).sample((sample_size,)).to(self.device)

    def __sample_negative(self, sample_size, rate):
        """
        Samples from rate^(-1) * Erlang(2, rate).
        :param sample_size: Number of samples
        :param rate: Rate parameter of the exponential distribution
        :return: Negative samples for the MVD of an exponential distribution.
        """
        k = 2
        uniform_samples = torch.rand((sample_size, k, *rate.shape), requires_grad=False).to(self.device)
        return - torch.log(uniform_samples.prod(dim=1))

    @staticmethod
    def __as_rate(params):
        log_rate, = params
        return torch.exp(log_rate)