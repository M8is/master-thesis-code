from typing import Tuple

import numpy as np
import scipy.stats
import torch

from .distribution_base import Distribution


class MultivariateNormal(Distribution):
    MIN_LOG_STD = -10
    MAX_LOG_STD = 6

    @staticmethod
    def param_dims(latent_dim: int) -> Tuple[int, ...]:
        return latent_dim, latent_dim

    def _as_params(self, raw_params):
        mean, log_std = torch.split(raw_params, raw_params.size(-1) // 2, dim=-1)
        log_std = torch.clip(log_std, MultivariateNormal.MIN_LOG_STD, MultivariateNormal.MAX_LOG_STD)
        std = torch.exp(log_std)
        return torch.stack((mean, std))

    def pdf(self):
        mean, std = self.params.cpu()
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

    def sample(self, sample_shape: torch.Size = torch.Size([])) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size([])) -> torch.Tensor:
        mean, std = self.params
        eps = torch.randn(sample_shape + mean.shape).to(self.device)
        return mean + eps * std

    def mvd_sample(self, size) -> torch.Tensor:
        mean, std = self.params
        mean_samples = self.__mean_samples(mean, std, size)
        std_samples = self.__std_samples(mean, std, size)
        return torch.stack((mean_samples, std_samples))

    def mvd_backward(self, losses, retain_graph):
        mean, std = self.params
        with torch.no_grad():
            losses = losses.mean(dim=2).transpose(-2, -1)  # Mean over samples
            mean_losses, std_losses = losses
            mean_pos_losses, mean_neg_losses = mean_losses
            mean_grad = (mean_pos_losses - mean_neg_losses) / (np.sqrt(2 * np.pi) * std)
            std_pos_losses, std_neg_losses = std_losses
            std_grad = (std_pos_losses - std_neg_losses) / std
            mean_grad /= mean_grad.size(0)
            std_grad /= std_grad.size(0)
        mean.backward(gradient=mean_grad, retain_graph=True)
        std.backward(gradient=std_grad, retain_graph=retain_graph)
        return torch.cat((mean_grad, std_grad), dim=-1)

    def kl(self):
        mean, std = self.params
        return 0.5 * (mean.pow(2) + std.pow(2) - 2 * torch.log(std) - 1).sum(dim=1)

    def log_prob(self, value):
        mean, std = self.params
        dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std ** 2))
        return dist.log_prob(value)

    def __mean_samples(self, mean, std, sample_size, coupled=False):
        std_normal = torch.randn((sample_size, *mean.size())).to(self.device)
        std_normal = std_normal.unsqueeze(-2).repeat_interleave(mean.size(-1), dim=-2)
        pos_weibull = self.__sample_weibull(sample_size, mean.size())
        pos_samples = self.__replace_diagonal(std_normal, pos_weibull)
        neg_weibull = pos_weibull if coupled else self.__sample_weibull(sample_size, mean.size())
        neg_samples = -self.__replace_diagonal(std_normal, neg_weibull)
        samples = torch.stack((pos_samples, neg_samples)).transpose(-3, -2)
        return mean + samples * std

    def __std_samples(self, mean, std, sample_size, coupled=True):
        std_normal = torch.randn((sample_size, *mean.size())).to(self.device)
        std_normal = std_normal.unsqueeze(-2).repeat_interleave(mean.size(-1), dim=-2)
        pos_standard_dsmaxwell = self.__sample_standard_doublesided_maxwell(sample_size, std.shape)
        pos_samples = self.__replace_diagonal(std_normal, pos_standard_dsmaxwell)
        if coupled:
            neg_std_dsmaxwell = self.__sample_standard_gaussian_from_standard_dsmaxwell(pos_standard_dsmaxwell)
        else:
            neg_std_dsmaxwell = torch.randn((sample_size, *mean.size(), mean.size(-1))).to(self.device)
        neg_samples = self.__replace_diagonal(std_normal, neg_std_dsmaxwell)
        samples = torch.stack((pos_samples, neg_samples)).transpose(-3, -2)
        return mean + std * samples

    @staticmethod
    def __replace_diagonal(target, new_diagonal) -> torch.Tensor:
        mask = 1. - torch.ones_like(new_diagonal).diag_embed()
        return mask * target + new_diagonal.diag_embed()

    def __sample_weibull(self, sample_size, shape):
        weibull = torch.distributions.Weibull(scale=np.sqrt(2.), concentration=2.)
        return weibull.sample((sample_size, *shape)).to(self.device)

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
