import torch

from . import base
from .mvd_distributions import normal, exponential, poisson


class MultivariateNormalMVD(base.MultivariateNormalProbabilistic):
    def grad_samples(self, params):
        mean, log_std = params
        std = torch.exp(log_std)
        mean_samples = normal.mean_samples(self.sample_size, mean, std)
        cov_samples = normal.cov_samples(self.sample_size, std)
        return torch.cat((mean_samples, cov_samples))

    def backward(self, params, losses):
        mean, std = params
        with torch.no_grad():
            mean_losses, std_losses = torch.split(losses, len(losses) // 2)
            mean_grad = self.__grad(normal.mean_constant(std), mean_losses)
            std_grad = self.__grad(normal.std_constant(std), std_losses)
        mean.backward(gradient=mean_grad, retain_graph=True)
        std.backward(gradient=std_grad)

    @staticmethod
    def __grad(c, losses):
        pos_losses, neg_losses = torch.split(losses, len(losses) // 2)
        delta = (pos_losses - neg_losses).view([-1] + list(c.shape))
        return c * delta.mean(dim=0)


class ExponentialMVD(base.ExponentialProbabilistic):
    def grad_samples(self, params):
        log_rate, = params
        return exponential.rate_samples(self.sample_size, torch.exp(log_rate))

    def backward(self, params, losses):
        rate, = params
        with torch.no_grad():
            mean_losses, std_losses = torch.split(losses, len(losses) // 2)
            rate_grad = self.__grad(rate, mean_losses)
        rate.backward(gradient=rate_grad)

    @staticmethod
    def __grad(rate, losses):
        pos_losses, neg_losses = torch.split(losses, len(losses) // 2)
        delta = (pos_losses - neg_losses).view([-1] + list(rate.shape))
        return exponential.rate_constant(rate) * delta.mean(dim=0)


class PoissonMVD(base.PoissonProbabilistic):
    def grad_samples(self, params):
        log_rate, = params
        return poisson.rate_samples(self.sample_size, torch.exp(log_rate))

    def backward(self, params, losses):
        rate, = params
        with torch.no_grad():
            mean_losses, std_losses = torch.split(losses, len(losses) // 2)
            rate_grad = self.__grad(rate.shape, mean_losses)
        rate.backward(gradient=rate_grad)

    @staticmethod
    def __grad(shape, losses):
        pos_losses, neg_losses = torch.split(losses, len(losses) // 2)
        delta = (pos_losses - neg_losses).view([-1] + list(shape))
        return delta.mean(dim=0)
