from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor


class Distribution(ABC):
    def __init__(self, output_size, device):
        self.param_dims = self._get_param_dims(output_size)
        self.latent_dim = output_size
        self.device = device
        self.retain_grad = False

    def retain_grad(self):
        self.retain_grad = True

    @abstractmethod
    def _get_param_dims(self, output_dim):
        pass

    @abstractmethod
    def mvd_backward(self, raw_params, losses, retain_graph) -> Tensor:
        pass

    @abstractmethod
    def sample(self, raw_params, size=1, with_grad=False) -> Tuple[Tensor, Tensor]:
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

    def pdf(self, raw_params):
        raise NotImplemented(f"PDF is not yet implemented for {type(self).__name__}")
