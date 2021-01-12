import torch


class Probabilistic:
    def __init__(self, sample_size):
        self.sample_size = sample_size

    @staticmethod
    def sample(params, size=1):
        raise NotImplementedError

    @staticmethod
    def kl(params):
        raise NotImplementedError

    def grad_samples(self, params):
        raise NotImplementedError

    def backward(self, params, losses):
        raise NotImplementedError


class MultivariateNormalProbabilistic(Probabilistic):
    @staticmethod
    def sample(params, size=1, with_grad=False):
        mean, log_std = params
        eps = torch.randn([size] + list(mean.shape), requires_grad=with_grad)
        return mean + eps * torch.exp(log_std)

    @staticmethod
    def kl(params):
        mean, log_std = params
        log_cov = 2 * log_std
        return 0.5 * (mean ** 2 + torch.exp(log_cov) - 1 - log_cov).sum(dim=1)

    def grad_samples(self, params):
        raise NotImplementedError

    def backward(self, params, losses):
        raise NotImplementedError
