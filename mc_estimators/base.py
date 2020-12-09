import torch


class Probabilistic:
    def __init__(self, sample_size, *_, **__):
        self.sample_size = sample_size
        self.__saved = None

    def _to_backward(self, value):
        self.__saved = value

    def _from_forward(self):
        value = self.__saved
        self.__saved = None
        return value

    @staticmethod
    def sample(params, size=1):
        raise NotImplementedError

    def grad_samples(self, params):
        raise NotImplementedError

    def backward(self, losses):
        raise NotImplementedError


class MultivariateNormalProbabilistic(Probabilistic):
    @staticmethod
    def sample(params, size=1):
        mean, log_std = params
        cov = torch.diag_embed(torch.exp(2 * log_std))
        return torch.distributions.MultivariateNormal(mean, cov).sample((size,))

    def grad_samples(self, params):
        raise NotImplementedError

    def backward(self, losses):
        raise NotImplementedError
