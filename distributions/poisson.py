import math

import torch
from .exponential import Exponential
from scipy.special import factorial


class Poisson(Exponential):
    def pdf(self, end: int = 20):
        with torch.no_grad():
            range_ = torch.arange(end)
            e = torch.tensor(math.e)
            pdf = self.params.pow(range_) * e.pow(-self.params) / factorial(range_)
            return range_, pdf.squeeze()

    def sample(self, sample_shape: torch.Size = torch.Size([])):
        dist = torch.distributions.poisson.Poisson(self.params)
        return dist.sample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size([])):
        raise NotImplementedError("Poisson cannot be reparameterized.")

    def mvsample(self, size):
        with torch.no_grad():
            pos_samples = self.__sample_poisson(size, self.params + 1)
            neg_samples = self.__sample_poisson(size, self.params)
            return torch.diag_embed(torch.stack((pos_samples, neg_samples)))

    def log_prob(self, value):
        return torch.distributions.Poisson(self.params).log_prob(value)

    @staticmethod
    def __sample_poisson(sample_size, rate):
        return torch.distributions.poisson.Poisson(rate).sample((sample_size,))
