import torch


def rate_samples(sample_size, rate):
    pos_samples = __sample_poisson(sample_size, rate + 1.)
    neg_samples = __sample_poisson(sample_size, rate)
    return torch.cat((pos_samples, neg_samples))


def __sample_poisson(sample_size, rate):
    return torch.distributions.poisson.Poisson(rate).sample(sample_size)
