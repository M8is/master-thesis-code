import torch
from .distribution_base import Distribution


class Exponential(Distribution):
    def _as_params(self, raw_params):
        return torch.exp(raw_params)

    def sample(self, sample_shape: torch.Size = torch.Size([])):
        dist = torch.distributions.exponential.Exponential(self.params)
        return dist.sample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size([])):
        dist = torch.distributions.exponential.Exponential(self.params)
        return dist.rsample(sample_shape)

    def mvsample(self, size):
        pos_samples = self.__sample_exponential(size, self.params)
        neg_samples = self.__sample_negative(size, self.params)
        samples = torch.diag_embed(torch.stack((pos_samples, neg_samples))).transpose(2, 3)
        return samples + self.params

    def mvd_backward(self, losses, retain_graph):
        with torch.no_grad():
            pos_losses, neg_losses = losses.mean(dim=1).transpose(-2, -1)  # Mean over samples
            grad = (pos_losses - neg_losses) / self.params
        assert grad.shape == self.params.shape, f"Grad shape {grad.shape} != params shape {self.params.shape}"
        self.params.backward(gradient=grad, retain_graph=retain_graph)

    def kl(self):
        return (self.params - torch.log(self.params) - 1).sum(dim=1)

    def log_prob(self, value):
        return torch.distributions.Exponential(self.params).log_prob(value)

    @staticmethod
    def __sample_exponential(sample_size, rate):
        return torch.distributions.exponential.Exponential(rate).sample((sample_size,))

    @staticmethod
    def __sample_negative(sample_size, rate):
        """
        Samples from rate^(-1) * Erlang(2, rate).
        :param sample_size: Number of samples
        :param rate: Rate parameter of the exponential distribution
        :return: Negative samples for the MVD of an exponential distribution.
        """
        k = 2
        uniform_samples = torch.rand((sample_size, k, *rate.shape), requires_grad=False).to(rate.device)
        return - torch.log(uniform_samples.prod(dim=1))
