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
        mean, std = self.__prepare(raw_params.cpu())
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
        return torch.stack((pos_samples, neg_samples))

    def mvd_backward(self, raw_params, losses, retain_graph):
        mean, std = self.__prepare(raw_params)
        with torch.no_grad():
            pos_losses, neg_losses = losses
            mean_pos_losses, std_pos_losses = pos_losses
            mean_neg_losses, std_neg_losses = neg_losses
            mean_c = self.__mean_c(std)
            std_c = self.__std_c(std)
            mean_grad = mean_c * (mean_pos_losses - mean_neg_losses)
            std_grad = std_c * (std_pos_losses - std_neg_losses)
        assert mean_grad.shape == mean.shape, f"Grad shape {mean_grad.shape} != params shape {mean.shape}"
        assert std_grad.shape == std.shape, f"Grad shape {std_grad.shape} != params shape {std.shape}"
        mean.backward(gradient=mean_grad, retain_graph=True)
        std.backward(gradient=std_grad, retain_graph=retain_graph)

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
        return (mean, log_std) if keep_log_std else (mean, torch.exp(log_std))

    def __mean_samples(self, sample_size, mean, std, coupled=False):
        pos_std_normal = torch.randn((sample_size, *mean.size(), mean.size(-1))).to(self.device)
        pos_weibull = self.__sample_weibull(sample_size, mean.size())
        pos_samples = self.__replace_diagonal(pos_std_normal, pos_weibull)
        neg_std_normal = torch.randn((sample_size, *mean.size(), mean.size(-1))).to(self.device)
        # TODO: Is this correct coupling?
        neg_weibull = pos_weibull if coupled else self.__sample_weibull(sample_size, mean.size())
        neg_samples = self.__replace_diagonal(neg_std_normal, neg_weibull)
        samples = torch.stack((pos_samples, -neg_samples)).transpose(-3, -2)
        return mean + samples * std

    def __std_samples(self, sample_size, mean, std, coupled=True):
        pos_std_normal = torch.randn((sample_size, *mean.size(), mean.size(-1))).to(self.device)
        pos_maxwell = self.__sample_standard_doublesided_maxwell(sample_size, std.shape)
        pos_samples = self.__replace_diagonal(pos_std_normal, pos_maxwell)
        neg_std_normal = torch.randn((sample_size, *mean.size(), mean.size(-1))).to(self.device)
        neg_maxwell = self.__sample_standard_gaussian_from_standard_dsmaxwell(pos_maxwell) if coupled \
            else torch.randn((sample_size, *mean.size(), mean.size(-1))).to(self.device)
        neg_samples = self.__replace_diagonal(neg_std_normal, neg_maxwell)
        samples = torch.stack((pos_samples, neg_samples)).transpose(-3, -2)
        return mean + std * samples

    @staticmethod
    def __replace_diagonal(target, new_diagonal):
        mask = torch.diag_embed(torch.ones_like(new_diagonal))
        return new_diagonal.diag_embed() + (1. - mask) * target

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

    @staticmethod
    def __mean_c(std):
        return 1. / (np.sqrt(2 * np.pi) * std)

    @staticmethod
    def __std_c(std):
        return 1. / std
