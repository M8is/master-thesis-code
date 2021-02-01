import math

import torch


class Probabilistic:
    def __init__(self, sample_size):
        self.sample_size = sample_size

    @staticmethod
    def sample(params, size=1, with_grad=False):
        raise NotImplementedError

    @staticmethod
    def kl(params):
        raise NotImplementedError

    def grad_samples(self, params):
        raise NotImplementedError

    def backward(self, params, losses):
        raise NotImplementedError


class MultivariateNormalProbabilistic(Probabilistic):
    @staticmethod
    def sample(params, size=1, with_grad=False):
        mean, log_std = params
        eps = torch.randn([size] + list(mean.shape), requires_grad=with_grad)
        return mean + eps * torch.exp(log_std)

    @staticmethod
    def kl(params):
        mean, log_std = params
        log_cov = 2 * log_std
        return 0.5 * (mean ** 2 + torch.exp(log_cov) - 1 - log_cov).sum(dim=1)

    def grad_samples(self, params):
        raise NotImplementedError

    def backward(self, params, losses):
        raise NotImplementedError


class ExponentialProbabilistic(Probabilistic):
    @staticmethod
    def sample(params, size=1, with_grad=False):
        log_rate, = params
        dist = torch.distributions.exponential.Exponential(torch.exp(log_rate))
        return dist.rsample(size) if with_grad else dist.sample(size)

    @staticmethod
    def kl(params):
        log_rate, = params
        return torch.exp(log_rate) - log_rate - 1

    def grad_samples(self, params):
        raise NotImplementedError

    def backward(self, params, losses):
        raise NotImplementedError


class PoissonProbabilistic(ExponentialProbabilistic):
    @staticmethod
    def sample(params, size=1, with_grad=False):
        log_rate, = params
        dist = torch.distributions.poisson.Poisson(torch.exp(log_rate))
        return dist.rsample(size) if with_grad else dist.sample(size)

    def grad_samples(self, params):
        raise NotImplementedError

    def backward(self, params, losses):
        raise NotImplementedError
