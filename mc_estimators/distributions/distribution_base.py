from abc import ABC, abstractmethod


class Distribution(ABC):
    def __init__(self, param_dims, device):
        self.param_dims = param_dims
        self.device = device

    @abstractmethod
    def mvd_backward(self, raw_params, losses, retain_graph):
        pass

    def pdf(self, raw_params):
        raise NotImplemented(f"PDF is not yet implemented for {type(self).__name__}")

    @abstractmethod
    def sample(self, raw_params, size=1, with_grad=False):
        pass

    @abstractmethod
    def mvd_sample(self, raw_params, size):
        pass

    @abstractmethod
    def kl(self, raw_params):
        pass

    @abstractmethod
    def log_prob(self, raw_params, samples):
        pass
