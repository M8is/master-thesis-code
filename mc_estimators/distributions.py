import torch
from abc import ABC, abstractmethod

from .mvd_distributions import normal, exponential, poisson


class Distribution(ABC):
    @staticmethod
    @abstractmethod
    def sample(params, size=1, with_grad=False):
        pass

    @staticmethod
    @abstractmethod
    def mvd_sample(params, size):
        pass

    @staticmethod
    @abstractmethod
    def mvd_grad(params, losses):
        pass

    @staticmethod
    @abstractmethod
    def kl(params):
        pass

    @staticmethod
    @abstractmethod
    def log_prob(params, samples):
        pass


class MultivariateNormalDistribution(Distribution):
    @staticmethod
    def sample(params, size=1, with_grad=False):
        mean, log_std = params
        eps = torch.randn([size] + list(mean.shape), requires_grad=with_grad)
        return mean + eps * torch.exp(log_std)

    @staticmethod
    def mvd_sample(params, size):
        mean, log_std = params
        std = torch.exp(log_std)
        mean_samples = normal.mean_samples(size, mean, std)
        cov_samples = normal.cov_samples(size, std)
        return torch.stack((mean_samples, cov_samples))

    @staticmethod
    def mvd_grad(params, losses):
        mean, std = params
        with torch.no_grad():
            mean_losses, std_losses = losses
            # Calculate mean grad
            pos_losses, neg_losses = torch.split(mean_losses, len(mean_losses) // 2)
            delta = (pos_losses - neg_losses).view([-1] + list(mean.shape))
            mean_grad = normal.mean_constant(std) * delta.mean(dim=0)
            # Calculate std grad
            pos_losses, neg_losses = torch.split(std_losses, len(std_losses) // 2)
            delta = (pos_losses - neg_losses).view([-1] + list(std.shape))
            std_grad = normal.std_constant(std) * delta.mean(dim=0)
        return torch.stack(mean_grad, std_grad)

    @staticmethod
    def kl(params):
        mean, log_std = params
        log_cov = 2 * log_std
        return 0.5 * (mean ** 2 + torch.exp(log_cov) - 1 - log_cov).sum(dim=1)

    @staticmethod
    def log_prob(params, samples):
        mean, log_std = params
        dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(torch.exp(2 * log_std)))
        return dist.log_prob(samples)


class ExponentialDistribution(Distribution):
    @staticmethod
    def sample(params, size=1, with_grad=False):
        log_rate, = params
        dist = torch.distributions.exponential.Exponential(torch.exp(log_rate))
        return dist.rsample(size) if with_grad else dist.sample(size)

    @staticmethod
    def mvd_sample(params, size):
        log_rate, = params
        return exponential.rate_samples(size, torch.exp(log_rate))

    @staticmethod
    def mvd_grad(params, losses):
        rate, = params
        with torch.no_grad():
            pos_losses, neg_losses = torch.split(losses, len(losses) // 2)
            delta = (pos_losses - neg_losses).view([-1] + list(rate.shape))
            rate_grad = exponential.rate_constant(rate) * delta.mean(dim=0)
        return rate_grad

    @staticmethod
    def kl(params):
        log_rate, = params
        return torch.exp(log_rate) - log_rate - 1

    @staticmethod
    def log_prob(params, samples):
        log_rate, = params
        return torch.distributions.Exponential(torch.exp(log_rate)).log_prob(samples)


class PoissonDistribution(ExponentialDistribution):
    @staticmethod
    def sample(params, size=1, with_grad=False):
        log_rate, = params
        dist = torch.distributions.poisson.Poisson(torch.exp(log_rate))
        return dist.rsample(size) if with_grad else dist.sample(size)

    @staticmethod
    def mvd_sample(params, size):
        log_rate, = params
        return poisson.rate_samples(size, torch.exp(log_rate))

    @staticmethod
    def mvd_grad(params, losses):
        rate, = params
        with torch.no_grad():
            pos_losses, neg_losses = torch.split(losses, len(losses) // 2)
            delta = (pos_losses - neg_losses).view([-1] + list(rate.shape))
            rate_grad = delta.mean(dim=0)  # c is 1 for Poisson distribution.
        return rate_grad

    @staticmethod
    def log_prob(params, samples):
        log_rate, = params
        return torch.distributions.Poisson(torch.exp(log_rate)).log_prob(samples)
