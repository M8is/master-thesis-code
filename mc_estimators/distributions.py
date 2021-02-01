from abc import ABC, abstractmethod

import torch

from mc_estimators.mvd_distributions import exponential, normal, poisson


class Distribution(ABC):
    @abstractmethod
    def sample(self, params, size=1, with_grad=False):
        pass

    @abstractmethod
    def mvd_sample(self, params, size):
        pass

    @abstractmethod
    def mvd_constant(self, params):
        pass

    @abstractmethod
    def kl(self, params):
        pass

    @abstractmethod
    def log_prob(self, params, samples):
        pass


class MultivariateNormal(Distribution):
    def sample(self, params, size=1, with_grad=False):
        mean, log_std = params
        eps = torch.randn([size] + list(mean.shape), requires_grad=with_grad)
        return mean + eps * torch.exp(log_std)

    def mvd_sample(self, params, size):
        mean, log_std = params
        std = torch.exp(log_std)
        mean_samples = normal.mean_samples(size, mean, std)
        cov_samples = normal.cov_samples(size, std)
        return torch.stack((mean_samples, cov_samples))

    def mvd_constant(self, params):
        mean, log_std = params
        std = torch.exp(log_std)
        return torch.stack((normal.mean_constant(std), normal.std_constant(std)))

    def kl(self, params):
        mean, log_std = params
        log_cov = 2 * log_std
        return 0.5 * (mean ** 2 + torch.exp(log_cov) - 1 - log_cov).sum(dim=1)

    def log_prob(self, params, samples):
        mean, log_std = params
        cov = torch.exp(2 * log_std)
        dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(cov))
        return dist.log_prob(samples)


class Exponential(Distribution):
    def sample(self, params, size=1, with_grad=False):
        log_rate, = params
        rate = torch.exp(log_rate)
        dist = torch.distributions.exponential.Exponential(rate)
        return dist.rsample((size,)) if with_grad else dist.sample((size,))

    def mvd_sample(self, params, size):
        log_rate, = params
        rate = torch.exp(log_rate)
        return exponential.rate_samples(size, rate)

    def mvd_constant(self, params):
        log_rate, = params
        rate = torch.exp(log_rate)
        return exponential.rate_constant(rate)

    def kl(self, params):
        log_rate, = params
        rate = torch.exp(log_rate)
        return (rate - log_rate - 1).sum(dim=1)

    def log_prob(self, params, samples):
        log_rate, = params
        rate = torch.exp(log_rate)
        return torch.distributions.Exponential(rate).log_prob(samples)


class Poisson(Exponential):
    def sample(self, params, size=1, with_grad=False):
        log_rate, = params
        rate = torch.exp(log_rate)
        if with_grad:
            return torch.poisson(rate.expand([size] + list(rate.shape)))
        else:
            dist = torch.distributions.poisson.Poisson(rate)
            return dist.sample((size,))

    def mvd_sample(self, params, size):
        log_rate, = params
        rate = torch.exp(log_rate)
        return poisson.rate_samples(size, rate)

    def log_prob(self, params, samples):
        log_rate, = params
        rate = torch.exp(log_rate)
        return torch.distributions.Poisson(rate).log_prob(samples)
