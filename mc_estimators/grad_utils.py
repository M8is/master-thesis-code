import torch


def get_random_cov(size):
    rand = torch.randn(size, size)
    return torch.nn.Parameter(rand @ rand.T)


def doublesided_maxwell(shape):
    gamma_sample = gamma(1.5, 0.5, shape)
    binomial_sample = binomial(1, 0.5, shape)
    dsmaxwell_sample = sqrt(gamma_sample) * (2 * binomial_sample - 1)
    return dsmaxwell_sample
