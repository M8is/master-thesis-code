from abc import ABC, abstractmethod


class Distribution(ABC):
    def __init__(self, param_dims, device):
        self.param_dims = param_dims
        self.device = device

    def mvd_grad(self, params, losses):
        pos_losses, neg_losses = losses
        delta = (pos_losses - neg_losses).mean(dim=1).transpose(-2, -1)
        c = self._mvd_constant(params)
        grad = c * delta
        if len(params) <= 1:
            grad.unsqueeze_(0)
        return grad

    @abstractmethod
    def sample(self, params, size=1, with_grad=False):
        pass

    @abstractmethod
    def mvd_sample(self, params, size):
        pass

    @abstractmethod
    def kl(self, params):
        pass

    @abstractmethod
    def log_prob(self, params, samples):
        pass

    @abstractmethod
    def _mvd_constant(self, params):
        pass
