import numpy as np
import scipy.stats
import torch

from .distribution_base import Distribution


class MultivariateNormal(Distribution):
    def sample(self, raw_params, size=1, with_grad=False):
        params = self.__prepare(raw_params)
        mean, log_std = params
        eps = torch.randn([size] + list(mean.shape), requires_grad=with_grad).to(self.device)
        return params, mean + eps * torch.exp(log_std)

    def mvd_sample(self, raw_params, size):
        params = self.__prepare(raw_params)
        mean, log_std = params
        std = torch.exp(log_std)
        mean_pos_samples, mean_neg_samples = self.__mean_samples(size, mean, std)
        cov_pos_samples, cov_neg_samples = self.__cov_samples(size, std)
        pos_samples = torch.stack((mean_pos_samples, cov_pos_samples))
        neg_samples = torch.stack((mean_neg_samples, cov_neg_samples))
        return params, torch.stack((mean, log_std)), torch.stack((pos_samples, neg_samples))

    def pdf(self, params):
        mean, log_std = self.__prepare(params)
        x = torch.tensor(np.linspace(-1.5, 2.5, 200))
        return x, scipy.stats.norm.pdf(x, mean, torch.exp(log_std))

    def _mvd_constant(self, params):
        _, log_std = params
        std = torch.exp(log_std)
        return torch.stack((self.__mean_constant(std), self.__std_constant(std)))

    def kl(self, params):
        mean, log_std = params
        log_cov = 2 * log_std
        return 0.5 * (mean ** 2 + torch.exp(log_cov) - 1 - log_cov).sum(dim=1)

    def log_prob(self, params, samples):
        mean, log_std = params
        cov = torch.exp(2 * log_std)
        dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(cov))
        return dist.log_prob(samples)

    def __prepare(self, params):
        return torch.stack(torch.split(params, self.param_dims, dim=-1))

    def __mean_samples(self, sample_size, mean, std, coupled=True):
        pos_samples = self.__sample_weibull(sample_size, mean.shape)
        neg_samples = pos_samples if coupled else self.__sample_weibull(sample_size, mean.shape)
        samples = torch.diag_embed(torch.stack((pos_samples, -neg_samples)) * std).transpose(2, 3)
        return samples + mean

    def __cov_samples(self, sample_size, std, coupled=True):
        pos_samples = self.__sample_standard_doublesided_maxwell(sample_size, std.shape)
        neg_samples = self.__sample_standard_gaussian_from_standard_dsmaxwell(pos_samples) if coupled \
            else self.__sample_standard_normal(sample_size, std.shape)
        return torch.diag_embed(torch.stack((pos_samples, neg_samples))).transpose(2, 3)

    def __sample_weibull(self, sample_size, shape):
        return torch.distributions.Weibull(np.sqrt(2.), concentration=2.).sample((sample_size, *shape)).to(self.device)

    def __sample_standard_doublesided_maxwell(self, sample_size, shape):
        gamma_sample = torch.distributions.Gamma(1.5, 0.5).sample((sample_size, *shape)).to(self.device)
        binomial_sample = torch.distributions.Binomial(1, 0.5).sample((sample_size, *shape)).to(self.device)
        return torch.sqrt(gamma_sample) * (2 * binomial_sample - 1)

    def __sample_standard_gaussian_from_standard_dsmaxwell(self, std_dsmaxwell_samples):
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
        dist = torch.distributions.uniform.Uniform(low=0., high=1.)
        return dist.sample(std_dsmaxwell_samples.shape).to(self.device) * std_dsmaxwell_samples

    def __sample_standard_normal(self, sample_size, shape):
        return torch.distributions.Normal(0, 1).sample((sample_size, *shape)).to(self.device)

    @staticmethod
    def __mean_constant(std):
        return 1. / (np.sqrt(2 * np.pi) * std)

    @staticmethod
    def __std_constant(std):
        return 1. / std
