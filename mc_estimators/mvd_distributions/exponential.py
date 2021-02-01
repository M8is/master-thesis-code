import torch


def rate_samples(sample_size, rate):
    pos_samples = __sample_exponential(sample_size, rate)
    neg_samples = __sample_negative(sample_size, rate)
    return torch.cat((pos_samples, neg_samples))


def rate_constant(rate):
    return 1. / rate


def __sample_exponential(sample_size, rate):
    return torch.distributions.exponential.Exponential(rate).sample(sample_size)


def __sample_negative(sample_size, rate):
    """
    Samples from rate^(-1) * Erlang(2, rate).
    :param sample_size: Number of samples
    :param rate: Rate parameter of the exponential distribution
    :return: Negative samples for the MVD of an exponential distribution.
    """
    k = 2
    uniform_samples = torch.rand([sample_size, k] + list(rate.shape), requires_grad=False)
    return - torch.log(uniform_samples.prod(dim=1))
