import numpy as np
import scipy.stats
import torch

from .distribution_base import Distribution


class MultivariateNormal(Distribution):
    MIN_LOG_STD = -10
    MAX_LOG_STD = 6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pdf(self, raw_params):
        mean, log_std = self.__prepare(raw_params.cpu())
        std = self.__exp(log_std)
        dims = mean.size(-1)
        if dims == 1:
            linspace = np.linspace(-3, 3, 300)
            return linspace, scipy.stats.norm.pdf(linspace, mean, std)
        elif dims == 2:
            x, y = np.mgrid[-3:3:.05, -3:3:.05]
            grid = np.dstack((x, y))
            rv = scipy.stats.multivariate_normal(mean, std.diag_embed())
            return (x, y), rv.pdf(grid)
        else:
            raise ValueError(f"Plotting is not supported for {dims}-dimensional gaussians.")

    def sample(self, raw_params, size=1, with_grad=False):
        mean, std = self.__prepare(raw_params)
        eps = torch.randn([size] + list(mean.shape), requires_grad=with_grad).to(self.device)
        return mean + eps * std

    def mvd_sample(self, raw_params, size):
        mean, std = self.__prepare(raw_params)
        mean_pos_samples, mean_neg_samples = self.__mean_samples(size, mean, std)
        cov_pos_samples, cov_neg_samples = self.__std_samples(size, mean, std)
        pos_samples = torch.stack((mean_pos_samples, cov_pos_samples))
        neg_samples = torch.stack((mean_neg_samples, cov_neg_samples))
        return torch.stack((pos_samples, neg_samples)).transpose(-2, -3)

    def mvd_backward(self, raw_params, losses, retain_graph):
        params = self.__prepare(raw_params).mean(dim=1)
        with torch.no_grad():
            pos_losses, neg_losses = losses
            c = self.__mvd_c(params[1])
            grad = c * (pos_losses - neg_losses).mean(dim=1)
        assert grad.shape == params.shape, f"Grad shape {grad.shape} != params shape {params.shape}"
        params.backward(gradient=grad, retain_graph=retain_graph)

    def __mvd_c(self, std):
        return torch.stack((self.__mean_c(std), self.__std_c(std)))

    def kl(self, raw_params):
        mean, log_std = self.__prepare(raw_params, keep_log_std=True)
        log_cov = 2 * log_std
        kl = 0.5 * (mean ** 2 + torch.exp(log_cov) - 1 - log_cov)
        return kl.sum(dim=1) if len(kl.shape) > 1 else kl

    def log_prob(self, raw_params, samples):
        mean, std = self.__prepare(raw_params)
        dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std**2))
        return dist.log_prob(samples)

    def __prepare(self, params, keep_log_std=False):
        mean, log_std = torch.split(params, self.param_dims, dim=-1)
        log_std = torch.clip(log_std, MultivariateNormal.MIN_LOG_STD, MultivariateNormal.MAX_LOG_STD)
        if not keep_log_std:
            return torch.stack((mean, torch.exp(log_std)))
        else:
            return torch.stack((mean, log_std))

    def __mean_samples(self, sample_size, mean, std, coupled=True):
        pos_samples = self.__sample_weibull(sample_size, mean.shape)
        neg_samples = pos_samples if coupled else self.__sample_weibull(sample_size, mean.shape)
        samples = torch.diag_embed(torch.stack((pos_samples, -neg_samples)) * std)
        return samples + mean.unsqueeze(-1)

    def __std_samples(self, sample_size, mean, std, coupled=True):
        pos_samples = self.__sample_standard_doublesided_maxwell(sample_size, std.shape)
        neg_samples = self.__sample_standard_gaussian_from_standard_dsmaxwell(pos_samples) if coupled \
            else self.__sample_standard_normal(sample_size, std.shape)
        pos_samples = mean + std * pos_samples
        neg_samples = mean + std * neg_samples
        return torch.diag_embed(torch.stack((pos_samples, neg_samples)))

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
    def __mean_c(std):
        return 1. / (np.sqrt(2 * np.pi) * std)

    @staticmethod
    def __std_c(std):
        return 1. / std
