import torch

from numpy.random import weibull
from math import pi, sqrt
from .base import MultivariateNormalProbabilistic


class MultivariateNormalMVD(MultivariateNormalProbabilistic):
    def __init__(self, sample_size, coupled=False):
        super().__init__(sample_size)
        self.coupled = coupled

    def grad_samples(self, params):
        mean, log_cov = params
        cov = torch.diag(torch.exp(log_cov.squeeze()))
        self._to_backward((mean, cov))

        pos_samples = self.__sample_weibull(mean.size(-1))
        if self.coupled:
            neg_samples = pos_samples
        else:
            neg_samples = self.__sample_weibull(mean.size(-1))
        samples = torch.cat((pos_samples, -neg_samples))

        l = torch.cholesky(cov)
        return samples.view(-1, mean.size(-1)) @ l + mean

    def backward(self, losses):
        mean, cov = self._from_forward()
        mean_grad = self.__mean_grad(cov, losses)
        cov_grad = self.__cov_grad(mean, cov, losses)
        return mean.backward(mean_grad), cov.backward(cov_grad)

    def __mean_grad(self, cov, losses):
        pos_losses, neg_losses = torch.split(losses, len(losses) // 2)
        pos_losses = pos_losses.view(self.sample_size, -1).sum(dim=0)
        neg_losses = neg_losses.view(self.sample_size, -1).sum(dim=0)
        c = torch.inverse(sqrt(2 * pi) * cov)
        grad_estimate = ((pos_losses - neg_losses) @ c) / self.sample_size
        return grad_estimate.unsqueeze(0)

    def __cov_grad(self, mean, cov, losses):
        # TODO: implement this
        return torch.zeros_like(cov)

    def __sample_weibull(self, size):
        return torch.diag_embed(torch.Tensor(weibull(sqrt(2.), (self.sample_size, size))))

    def __sample_doublesided_maxwell(self, size):
        gamma_sample = torch.distributions.Gamma(1.5, 0.5).sample((self.sample_size, size))
        binomial_sample = torch.distributions.Binomial(1, 0.5).sample((self.sample_size, size))
        dsmaxwell_sample = torch.sqrt(gamma_sample) * (2 * binomial_sample - 1)
        return dsmaxwell_sample
