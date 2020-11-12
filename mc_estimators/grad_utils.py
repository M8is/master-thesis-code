import torch


def get_random_cov(size):
    rand = torch.randn(size, size)
    return torch.nn.Parameter(rand @ rand.T)


def doublesided_maxwell(shape):
    gamma_sample = torch.distributions.Gamma(1.5, 0.5).sample(shape)
    binomial_sample = torch.distributions.Binomial(1, 0.5).sample(shape)
    dsmaxwell_sample = torch.sqrt(gamma_sample) * (2 * binomial_sample - 1)
    return dsmaxwell_sample
