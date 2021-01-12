import torch

from torch.distributions import Weibull
from numpy import pi, sqrt
from .base import MultivariateNormalProbabilistic


class MultivariateNormalMVD(MultivariateNormalProbabilistic):
    def __init__(self, sample_size, coupled=False):
        super().__init__(sample_size)
        self.coupled = coupled

    def grad_samples(self, params):
        mean, log_std = params
        std = torch.exp(log_std)
        mean_samples = self.__mean_samples(mean, std)
        cov_samples = self.__cov_samples(std)
        return torch.cat((mean_samples, cov_samples))

    def backward(self, params, losses):
        mean, std = params
        with torch.no_grad():
            mean_losses, std_losses = torch.split(losses, len(losses) // 2)
            mean_grad = self.__mean_grad(std, mean_losses)
            std_grad = self.__std_grad(std, std_losses)
        mean.backward(gradient=mean_grad, retain_graph=True)
        std.backward(gradient=std_grad)

    def __mean_samples(self, mean, std):
        pos_samples = self.__sample_weibull(mean.shape)
        neg_samples = pos_samples if self.coupled else self.__sample_weibull(mean.shape)
        samples = torch.diag_embed(torch.cat((pos_samples, -neg_samples)) * std).view([-1] + list(mean.shape))
        return samples + mean

    def __cov_samples(self, std):
        pos_samples = self.__sample_standard_doublesided_maxwell(std.shape)
        neg_samples = self.__sample_standard_gaussian_from_standard_dsmaxwell(pos_samples) if self.coupled \
            else self.__sample_standard_normal(std.shape)
        return torch.diag_embed(torch.cat((pos_samples, neg_samples))).view([-1] + list(std.shape))

    def __mean_grad(self, std, losses):
        return self.__grad(1. / (sqrt(2 * pi) * std), losses)

    def __std_grad(self, std, losses):
        return self.__grad(1. / std, losses)

    def __grad(self, c, losses):
        pos_losses, neg_losses = torch.split(losses, len(losses) // 2)
        delta = (pos_losses - neg_losses).view([-1] + list(c.shape))
        return c * delta.mean(dim=0)

    def __sample_weibull(self, shape):
        return Weibull(sqrt(2.), concentration=2.).sample([self.sample_size] + list(shape))

    def __sample_standard_doublesided_maxwell(self, shape):
        gamma_sample = torch.distributions.Gamma(1.5, 0.5).sample([self.sample_size] + list(shape))
        binomial_sample = torch.distributions.Binomial(1, 0.5).sample([self.sample_size] + list(shape))
        dsmaxwell_sample = torch.sqrt(gamma_sample) * (2 * binomial_sample - 1)
        return dsmaxwell_sample

    @staticmethod
    def __sample_standard_gaussian_from_standard_dsmaxwell(std_dsmaxwell_samples):
        """
        Adapted from https://github.com/deepmind/mc_gradients
        Generate Gaussian variates from Double-sided Maxwell variates.

        Useful for coupling samples from Gaussian and Double-sided Maxwell dist.
        1. Generate ds-maxwell variates: dsM ~ dsMaxwell(0,1)
        2. Generate uniform variates: u ~ Unif(0,1)
        3. multiply y = u * dsM
        The result is Gaussian distribution N(0,1) which can be loc-scale adjusted.

        Args:
            std_dsmaxwell_samples: Samples generated from a zero-mean, unit variance
                                   double-sided Maxwell distribution M(0,1).
        Returns:
            Tensor of Gaussian variates with the same shape as the input.
        """
        uniform_rvs = torch.distributions.uniform.Uniform(low=0., high=1.).sample(std_dsmaxwell_samples.shape)
        return uniform_rvs * std_dsmaxwell_samples

    def __sample_standard_normal(self, shape):
        standard_normal = torch.distributions.Normal(0, 1)
        return standard_normal.sample([self.sample_size] + list(shape))
