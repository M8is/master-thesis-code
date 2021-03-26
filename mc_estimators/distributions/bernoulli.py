import torch
from .distribution_base import Distribution


class Bernoulli(Distribution):
    def sample(self, params, size=1, with_grad=False):
        return torch.distributions.Bernoulli(self.__as_prob(params)).sample((size,))

    def mvd_sample(self, params, size):
        return torch.tensor([1, 0]).reshape(2, 1, 1, 1, 1).to(self.device)

    def mvd_constant(self, params):
        return 1.

    def kl(self, params):
        p = self.__as_prob(params)
        return torch.log(torch.tensor(.5)) - .5 * (torch.log(p) + torch.log(1 - p))

    def log_prob(self, params, samples):
        return torch.distributions.Bernoulli(self.__as_prob(params)).log_prob(samples)

    @staticmethod
    def __as_prob(params):
        return torch.sigmoid(params[0])
