from abc import ABC, abstractmethod
from typing import Tuple

import torch


class Distribution(ABC):
    def __init__(self, raw_params, device):
        self.params = self._as_params(raw_params)
        self.device = device

    @staticmethod
    def param_dims(latent_dim: int) -> Tuple[int, ...]:
        return latent_dim,

    @abstractmethod
    def _as_params(self, raw_params):
        pass

    @property
    @abstractmethod
    def kl(self):
        pass

    @property
    def pdf(self):
        raise NotImplemented(f"PDF is not yet implemented for {type(self).__name__}")

    @abstractmethod
    def mvd_backward(self, losses, retain_graph) -> torch.Tensor:
        pass

    @abstractmethod
    def sample(self, sample_shape: torch.Size = torch.Size([])) -> torch.Tensor:
        pass

    @abstractmethod
    def rsample(self, sample_shape: torch.Size = torch.Size([])) -> torch.Tensor:
        pass

    @abstractmethod
    def mvd_sample(self, size: int) -> torch.Tensor:
        pass

    @abstractmethod
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        pass
