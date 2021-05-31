import torch
from .distribution_base import Distribution


class Exponential(Distribution):
    def sample(self, raw_params, size=1, with_grad=False):
        dist = torch.distributions.exponential.Exponential(self._as_rate(raw_params))
        samples = dist.rsample((size,)) if with_grad else dist.sample((size,))
        return samples.to(self.device)

    def mvd_sample(self, raw_params, size):
        rate = self._as_rate(raw_params)
        pos_samples = self.__sample_exponential(size, rate)
        neg_samples = self.__sample_negative(size, rate)
        samples = torch.diag_embed(torch.stack((pos_samples, neg_samples))).transpose(2, 3)
        return samples + rate

    def mvd_backward(self, raw_params, losses, retain_graph):
        rate = self._as_rate(raw_params)
        with torch.no_grad():
            pos_losses, neg_losses = losses.mean(dim=0)
            grad = (1. / rate) * (pos_losses - neg_losses)
        assert grad.shape == rate.shape, f"Grad shape {grad.shape} != params shape {rate.shape}"
        rate.backward(gradient=grad, retain_graph=retain_graph)

    def kl(self, raw_params):
        params = self._as_rate(raw_params)
        return (params - torch.log(params) - 1).sum(dim=1)

    def log_prob(self, raw_params, samples):
        params = self._as_rate(raw_params)
        return torch.distributions.Exponential(params).log_prob(samples).sum(dim=-1)

    @staticmethod
    def _as_rate(raw_params):
        return torch.exp(raw_params)

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
