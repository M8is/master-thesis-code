from abc import ABC, abstractmethod


class Distribution(ABC):
    def __init__(self, param_dims, device):
        self.param_dims = param_dims
        self.device = device

    def mvd_grad(self, params, losses):
        pos_losses, neg_losses = losses.unbind(dim=-2)
        delta = (pos_losses - neg_losses).mean(dim=0).permute(2, 0, 1)
        c = self._mvd_constant(params)
        grad = c * delta
        if len(params) <= 1:
            grad.unsqueeze_(0)
        return grad

    def pdf(self, params):
        raise NotImplemented(f"PDF is not yet implemented for {type(self).__name__}")

    @abstractmethod
    def sample(self, raw_params, size=1, with_grad=False):
        pass

    @abstractmethod
    def mvd_sample(self, raw_params, size):
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
