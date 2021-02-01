import numpy as np
import torch


def mean_samples(sample_size, mean, std, coupled=True):
    pos_samples = sample_weibull(sample_size, mean.shape)
    neg_samples = pos_samples if coupled else sample_weibull(sample_size, mean.shape)
    samples = torch.diag_embed(torch.cat((pos_samples, -neg_samples)) * std).view([-1] + list(mean.shape))
    return samples + mean


def cov_samples(sample_size, std, coupled=True):
    pos_samples = sample_standard_doublesided_maxwell(sample_size, std.shape)
    neg_samples = sample_standard_gaussian_from_standard_dsmaxwell(pos_samples) if coupled \
        else sample_standard_normal(sample_size, std.shape)
    return torch.diag_embed(torch.cat((pos_samples, neg_samples))).view([-1] + list(std.shape))


def mean_constant(std):
    return 1. / (np.sqrt(2 * np.pi) * std)


def std_constant(std):
    return 1. / std


def sample_weibull(sample_size, shape):
    return torch.distributions.Weibull(np.sqrt(2.), concentration=2.).sample([sample_size] + list(shape))


def sample_standard_doublesided_maxwell(sample_size, shape):
    gamma_sample = torch.distributions.Gamma(1.5, 0.5).sample([sample_size] + list(shape))
    binomial_sample = torch.distributions.Binomial(1, 0.5).sample([sample_size] + list(shape))
    dsmaxwell_sample = torch.sqrt(gamma_sample) * (2 * binomial_sample - 1)
    return dsmaxwell_sample


def sample_standard_gaussian_from_standard_dsmaxwell(std_dsmaxwell_samples):
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


def sample_standard_normal(sample_size, shape):
    standard_normal = torch.distributions.Normal(0, 1)
    return standard_normal.sample([sample_size] + list(shape))
