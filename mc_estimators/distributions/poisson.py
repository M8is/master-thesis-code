import torch
from .exponential import Exponential


class Poisson(Exponential):
    def sample(self, sample_shape: torch.Size = torch.Size([])):
        dist = torch.distributions.poisson.Poisson(self.params)
        return dist.sample(sample_shape).to(self.device)

    def rsample(self, sample_shape: torch.Size = torch.Size([])):
        raise ValueError("Poisson cannot be reparameterized.")

    def mvd_sample(self, size):
        with torch.no_grad():
            pos_samples = self.__sample_poisson(size, self.params + 1.)
            neg_samples = self.__sample_poisson(size, self.params)
            return torch.diag_embed(torch.stack((pos_samples, neg_samples))).transpose(2, 3)

    def log_prob(self, value):
        return torch.distributions.Poisson(self.params).log_prob(value).sum(dim=-1).to(self.device)

    def __sample_poisson(self, sample_size, rate):
        return torch.distributions.poisson.Poisson(rate).sample((sample_size,)).to(self.device)
