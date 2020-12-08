import torch

from torch.distributions import Weibull
from numpy import pi, sqrt
from .base import MultivariateNormalProbabilistic


class MultivariateNormalMVD(MultivariateNormalProbabilistic):
    def __init__(self, sample_size, coupled=False):
        super().__init__(sample_size)
        self.coupled = coupled

    def grad_samples(self, params):
        mean, std = params
        self._to_backward((mean, std))
        mean_samples = self.__mean_grad_samples(mean, std)
        cov_samples = self.__cov_grad_samples(std)
        return torch.cat((mean_samples, cov_samples))

    def backward(self, losses):
        with torch.no_grad():
            mean, std = self._from_forward()
            mean_losses, std_losses = torch.split(losses, len(losses) // 2)
            mean_grad = self.__mean_grad(std, mean_losses)
            std_grad = self.__std_grad(std, std_losses)
        return mean.backward(gradient=mean_grad, retain_graph=True), std.backward(gradient=std_grad)

    def __mean_grad_samples(self, mean, std):
        output_size = mean.size(-1)
        pos_samples = self.__sample_weibull(output_size)
        neg_samples = -pos_samples if self.coupled else -self.__sample_weibull(output_size)
        mean_samples = torch.cat((pos_samples, neg_samples))
        mean_samples = torch.diag_embed(mean_samples).view(-1, mean.size(-1)) @ std.T + mean
        return mean_samples

    @staticmethod
    def __mean_grad(std, losses):
        pos_losses, neg_losses = torch.split(losses, len(losses) // 2)
        pos_losses = pos_losses.view(1, -1, std.size(-1))
        neg_losses = neg_losses.view(1, -1, std.size(-1))
        c = 1. / (sqrt(2 * pi) * std.squeeze())
        delta = pos_losses - neg_losses
        return delta.mean(dim=1) * c

    def __cov_grad_samples(self, std):
        output_size = std.size(-1)
        pos_samples = self.__sample_standard_doublesided_maxwell(output_size)
        neg_samples = self.__sample_standard_gaussian_from_standard_dsmaxwell(pos_samples) if self.coupled \
            else self.__sample_standard_normal(output_size)
        cov_samples = torch.cat((pos_samples, neg_samples))
        cov_samples = torch.diag_embed(cov_samples).view(-1, output_size)
        return cov_samples

    @staticmethod
    def __std_grad(std, losses):
        pos_losses, neg_losses = torch.split(losses, len(losses) // 2)
        pos_losses = pos_losses.view(1, -1, std.size(-1))
        neg_losses = neg_losses.view(1, -1, std.size(-1))
        c = torch.diag((1. / std).squeeze())
        delta = pos_losses - neg_losses
        return delta.mean(dim=1) @ c

    def __sample_weibull(self, size):
        return Weibull(sqrt(2.), concentration=2.).sample((self.sample_size, size))

    def __sample_standard_doublesided_maxwell(self, size):
        gamma_sample = torch.distributions.Gamma(1.5, 0.5).sample((self.sample_size, size))
        binomial_sample = torch.distributions.Binomial(1, 0.5).sample((self.sample_size, size))
        dsmaxwell_sample = torch.sqrt(gamma_sample) * (2 * binomial_sample - 1)
        return dsmaxwell_sample

    @staticmethod
    def __sample_standard_gaussian_from_standard_dsmaxwell(std_dsmaxwell_samples):
        """
        Adapted from https://github.com/deepmind/mc_gradients
        Generate Gaussian variates from Double-sided Maxwell variates.

        Useful for coupling samples from Gaussian and double_sided Maxwell dist.
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
        uniform_rvs = torch.distributions.uniform.Uniform(low=0., high=1.).sample(std_dsmaxwell_samples.size())
        return uniform_rvs * std_dsmaxwell_samples

    def __sample_standard_normal(self, size):
        standard_normal = torch.distributions.MultivariateNormal(torch.zeros(size), torch.eye(size))
        return standard_normal.sample((self.sample_size,))
